"""
EEGNet Classifier with integrated training and data preparation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from classification.base_classifier import BaseEEGClassifier
from classification.models.eeg_net.eeg_net import EEGNet

class EEGNetClassifier(BaseEEGClassifier):
    """EEGNet classifier with integrated training and data preparation"""
    
    def __init__(self, lr=5e-4, epochs=75, batch_size=32, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, **model_params):
        super().__init__(name="EEGNet")
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.model_params = model_params
        self._setup_tf()
        
    def _setup_tf(self):
        """Setup TensorFlow GPU if available"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    
    def prepare_data(self, X_train, y_train, X_eval, y_eval, subject_ids=None):
        """Prepare data with per-subject normalization and reshaping"""
        def normalize_and_reshape(X, y, subject_ids=None):
            if subject_ids is not None:
                # Per-subject z-score normalization
                X_scaled = np.zeros_like(X)
                unique_subs = np.unique(subject_ids)
                
                for sub_id in unique_subs:
                    sub_indices = np.where(subject_ids == sub_id)[0]
                    sub_data = X[sub_indices]
                    sub_mean = np.mean(sub_data, axis=(0, 2), keepdims=True)
                    sub_std = np.std(sub_data, axis=(0, 2), keepdims=True)
                    X_scaled[sub_indices] = (sub_data - sub_mean) / (sub_std + 1e-6)
            else:
                # Global normalization if no subject IDs
                X_scaled = (X - np.mean(X)) / (np.std(X) + 1e-6)
            
            # Reshape for Keras (add channel dimension)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], X_scaled.shape[2], 1)
            return X_reshaped, np.array(y)
        
        # Process datasets
        X_train_prep, y_train_prep = normalize_and_reshape(X_train, y_train, subject_ids)
        X_eval_prep, y_eval_prep = normalize_and_reshape(X_eval, y_eval)
            
        return X_train_prep, y_train_prep, X_eval_prep, y_eval_prep
    
    def train(self, X_train, y_train, X_eval, y_eval, subject_ids=None, verbose=True):
        """Train EEGNet classifier"""
        if verbose:
            print(f"\n[EEGNet Training] LR: {self.lr}, Epochs: {self.epochs}, Dropout: {self.dropoutRate}")
        
        # Prepare data
        X_train_prep, y_train_prep, X_eval_prep, y_eval_prep = self.prepare_data(
            X_train, y_train, X_eval, y_eval, subject_ids)
        
        # Build model for binary classification
        nb_classes = 1  # Single sigmoid output for binary classification
        Chans = X_train_prep.shape[1]
        Samples = X_train_prep.shape[2]

        # Auto-adjust kernel length
        if self.kernLength == 64 and Samples > 400:
            self.kernLength = min(250, Samples // 4)

        self.model = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples,
                           dropoutRate=self.dropoutRate, kernLength=self.kernLength,
                           F1=self.F1, D=self.D, F2=self.F2, **self.model_params)

        # Calculate proper balanced class weights for imbalanced data
        class_counts = np.bincount(y_train_prep.astype(int))
        total_samples = len(y_train_prep)
        
        # Balanced class weights (more aggressive for minority class)
        class_weight = {
            0: total_samples / (2 * class_counts[0]),
            1: total_samples / (2 * class_counts[1])
        }
        
        if verbose:
            print(f"  Class distribution: {class_counts}")
            print(f"  Class weights: {class_weight}")
            minority_boost = class_weight[1] / class_weight[0]
            print(f"  Minority class boost: {minority_boost:.2f}x")

        # Compile with binary crossentropy, AUC metric, and lower learning rate
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0),  # Add gradient clipping
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Train with early stopping (monitor AUC to prevent collapse)
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max'),
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, mode='min')
        ]
        
        # Train model
        history = self.model.fit(
            X_train_prep, y_train_prep,
            validation_data=(X_eval_prep, y_eval_prep),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )
        
        # Check for model collapse
        val_probs = self.model.predict(X_eval_prep, verbose=0).flatten()
        prob_range = np.max(val_probs) - np.min(val_probs)
        
        if verbose:
            print(f"  Validation prob range: {prob_range:.4f} (>0.3 is healthy)")
            if prob_range < 0.1:
                print(f"  ⚠️  WARNING: Model might be collapsing (range < 0.1)")
        
        self.is_trained = True
        
        # Evaluate
        results = self.evaluate(X_eval_prep, y_eval_prep, "Evaluation")
        
        return results

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Prepare data (just reshape, no subject normalization for prediction)
        X_prep = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        probs = self.model.predict(X_prep, verbose=0).flatten()  # Single sigmoid output
        return (probs > 0.5).astype(int)  # Binary threshold
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_prep = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        probs = self.model.predict(X_prep, verbose=0).flatten()  # Single sigmoid output
        # Return probabilities for both classes [p(class=0), p(class=1)]
        return np.column_stack([1 - probs, probs])