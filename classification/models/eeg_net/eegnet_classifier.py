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
    
    def __init__(self, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, **model_params):
        super().__init__(name="EEGNet")
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
    
    def prepare_data(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, subject_ids=None):
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
        
        # Process all datasets
        X_train_prep, y_train_prep = normalize_and_reshape(X_train, y_train, subject_ids)
        X_val_prep, y_val_prep = normalize_and_reshape(X_val, y_val)
        
        if X_test is not None:
            X_test_prep, y_test_prep = normalize_and_reshape(X_test, y_test)
        else:
            X_test_prep, y_test_prep = None, None
            
        return X_train_prep, y_train_prep, X_val_prep, y_val_prep, X_test_prep, y_test_prep
    
    def train(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, 
              lr=1e-3, epochs=100, batch_size=64, subject_ids=None, verbose=True):
        """Train EEGNet classifier"""
        if verbose:
            print(f"\\n[EEGNet Training] LR: {lr}, Epochs: {epochs}, Dropout: {self.dropoutRate}")
        
        # Prepare data
        X_train_prep, y_train_prep, X_val_prep, y_val_prep, X_test_prep, y_test_prep = self.prepare_data(
            X_train, y_train, X_val, y_val, X_test, y_test, subject_ids)
        
        # Build model
        nb_classes = 2
        Chans = X_train_prep.shape[1]
        Samples = X_train_prep.shape[2]
        
        # Auto-adjust kernel length
        if self.kernLength == 64 and Samples > 400:
            self.kernLength = min(250, Samples // 4)
        
        self.model = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples,
                           dropoutRate=self.dropoutRate, kernLength=self.kernLength,
                           F1=self.F1, D=self.D, F2=self.F2, **self.model_params)
        
        # Compile
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=['accuracy']
        )
        
        # Train with early stopping
        callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
        
        history = self.model.fit(
            X_train_prep, y_train_prep,
            validation_data=(X_val_prep, y_val_prep),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )
        
        self.is_trained = True
        
        # Evaluate
        if X_test_prep is not None:
            results = self.evaluate(X_test_prep, y_test_prep, "Test")
        else:
            results = self.evaluate(X_val_prep, y_val_prep, "Validation")
        
        return results
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Prepare data (just reshape, no subject normalization for prediction)
        X_prep = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        probs = self.model.predict(X_prep, verbose=0)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_prep = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return self.model.predict(X_prep, verbose=0)