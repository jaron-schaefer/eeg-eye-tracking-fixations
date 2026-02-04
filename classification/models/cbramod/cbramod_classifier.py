"""
CBraMod Classifier with integrated training and data preparation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from einops.layers.torch import Rearrange
from scipy.signal import resample
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from classification.base_classifier import BaseEEGClassifier
from classification.models.cbramod.cbramod import CBraMod
from data.data_configs import EEG_SAMPLE_FREQUENCY
import os
import gc

class LabelSmoothingBCELoss(nn.Module):
    """BCE Loss with label smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing * 0.5
        return nn.functional.binary_cross_entropy_with_logits(inputs, smooth_targets)

class CBraModWrapper(nn.Module):
    """CBraMod with MLP head following paper recommendations"""
    def __init__(self, backbone, input_dim, dropout=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = backbone
        self.backbone.proj_out = nn.Identity()  # Remove reconstruction head
        
        # MLP head as recommended in paper
        self.classifier = nn.Sequential(
            Rearrange('b c s p -> b (c s p)'),  # Flatten representations
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # Non-linear activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Binary classification output
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights properly"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class CBraModClassifier(BaseEEGClassifier):
    """CBraMod classifier following paper recommendations for binary classification"""
    
    def __init__(self, lr=1e-4, epochs=75, batch_size=64, weight_decay=5e-2, dropout=0.1, hidden_dim=512, label_smoothing=0.0):
        super().__init__(name="CBraMod")
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.label_smoothing = label_smoothing  # Paper uses 0.0 for binary tasks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_data(self, X_train, y_train, X_eval, y_eval, 
                    subject_ids=None, orig_fs=EEG_SAMPLE_FREQUENCY):
        """Prepare data for CBraMod with resampling, filtering, and patching"""
        
        def process_data(X, y, subject_ids=None):
            # Convert to arrays
            X, y = np.array(X), np.array(y)
            if subject_ids is not None:
                subject_ids = np.array(subject_ids)
            
            # Resample to 200 Hz
            target_fs = 200
            if orig_fs != target_fs:
                n_times = X.shape[2]
                new_n_times = int(n_times * target_fs / orig_fs)
                X = resample(X, new_n_times, axis=2)
            
            # Remove bad trials (>100µV)
            valid_mask = np.all((X <= 100) & (X >= -100), axis=(1, 2))
            X_filtered = X[valid_mask]
            y_filtered = y[valid_mask]
            
            if len(X_filtered) == 0:
                raise ValueError("All data filtered out! Check signal units (should be µV)")
            
            # Scale by 100µV (CRITICAL: Paper requires this exact scaling)
            X_scaled = X_filtered / 100.0
            
            # Validate scaling (should be roughly in [-1, 1] range after scaling)
            if np.abs(X_scaled).max() > 10:
                print(f"⚠️  WARNING: Data may not be in µV units. Max value after /100: {np.abs(X_scaled).max():.2f}")
                print("   Expected: data should be in microvolts (µV) for CBraMod")
            elif np.abs(X_scaled).max() < 0.01:
                print(f"⚠️  WARNING: Data may be in wrong units. Max value after /100: {np.abs(X_scaled).max():.6f}")
                print("   Expected: data should be in microvolts (µV), not volts")
            
            # Patching: reshape to (batch, channels, patches, 200)
            batch_size, n_channels, n_times = X_scaled.shape
            n_patches = n_times // 200
            
            if n_patches == 0:
                raise ValueError(f"Data too short for CBraMod. Need at least 200 samples, got {n_times}")
            
            # Trim to fit patches
            X_trimmed = X_scaled[:, :, :n_patches * 200]
            X_patches = X_trimmed.reshape(batch_size, n_channels, n_patches, 200)
            
            return X_patches, y_filtered
        
        # Process all datasets
        X_train_prep, y_train_prep = process_data(X_train, y_train, subject_ids)
        X_eval_prep, y_eval_prep = process_data(X_eval, y_eval)
            
        return X_train_prep, y_train_prep, X_eval_prep, y_eval_prep
    
    def train(self, X_train, y_train, X_eval, y_eval, subject_ids=None, verbose=True):
        """Train CBraMod classifier following paper's binary classification approach"""
        if verbose:
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
            print(f"\n[CBraMod Training] Device: {gpu_info}, LR: {self.lr}, Epochs: {self.epochs}, Batch: {self.batch_size}")
            print(f"Following paper: Binary BCE loss, full fine-tuning, cosine annealing")
        
        # Prepare data
        X_train_prep, y_train_prep, X_eval_prep, y_eval_prep = self.prepare_data(
            X_train, y_train, X_eval, y_eval, subject_ids)
        
        # Check for class imbalance (common in target detection)
        class_counts = np.bincount(y_train_prep.astype(int))
        if len(class_counts) == 2:
            pos_ratio = class_counts[1] / class_counts[0] if class_counts[0] > 0 else 1.0
            if verbose:
                print(f"Class distribution: {class_counts[0]} non-targets, {class_counts[1]} targets")
                if pos_ratio < 0.2 or pos_ratio > 5.0:
                    print(f"⚠️  Class imbalance detected (ratio: {pos_ratio:.2f}). Using weighted loss.")
        
        # Create data loaders
        train_ds = TensorDataset(torch.from_numpy(X_train_prep).float(), 
                                torch.from_numpy(y_train_prep.astype(np.float32)))
        eval_ds = TensorDataset(torch.from_numpy(X_eval_prep).float(),
                              torch.from_numpy(y_eval_prep.astype(np.float32)))
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        eval_loader = DataLoader(eval_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        
        # Build model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, 'pretrained_weights.pth')
        
        backbone = CBraMod().to(self.device)
        try:
            backbone.load_state_dict(torch.load(weights_path, map_location=self.device))
            if verbose:
                print("✅ Loaded pretrained weights")
        except FileNotFoundError:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
        
        # Calculate input dimension
        _, n_channels, n_patches, n_time = X_train_prep.shape
        input_dim = n_channels * n_patches * n_time
        
        self.model = CBraModWrapper(backbone, input_dim, self.dropout, self.hidden_dim).to(self.device)
        
        # Training setup following paper's binary classification approach
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Calculate pos_weight for class imbalance (paper emphasizes balanced metrics)
        class_counts = np.bincount(y_train_prep.astype(int))
        if len(class_counts) == 2 and class_counts[0] > 0:
            pos_weight = torch.tensor(class_counts[0] / class_counts[1]).to(self.device)
        else:
            pos_weight = torch.tensor(1.0).to(self.device)
        
        # Loss function: Standard BCE for binary (paper approach) or label smoothing if specified
        if self.label_smoothing > 0:
            criterion = LabelSmoothingBCELoss(smoothing=self.label_smoothing)
            if verbose:
                print(f"Using label smoothing ({self.label_smoothing}) - not recommended by paper for binary tasks")
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Weighted BCE for imbalance
            if verbose and pos_weight.item() != 1.0:
                print(f"Using weighted BCE loss (pos_weight: {pos_weight.item():.2f}) for class imbalance")
            
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Cosine annealing scheduler as recommended
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.lr*0.01)
        
        # Early stopping
        best_eval_acc = 0.0
        best_weights = None
        patience = 0
        patience_limit = 20  # Increased for 50 epochs
        
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(X_batch).squeeze()
                        loss = criterion(logits, y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(X_batch).squeeze()
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    train_preds.extend((probs > 0.5).float().cpu().numpy())
                    train_labels.extend(y_batch.cpu().numpy())
            
            # Evaluation phase
            self.model.eval()
            eval_preds, eval_labels = [], []
            
            with torch.no_grad():
                for X_batch, y_batch in eval_loader:
                    X_batch = X_batch.to(self.device)
                    
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            logits = self.model(X_batch).squeeze()
                    else:
                        logits = self.model(X_batch).squeeze()
                    
                    probs = torch.sigmoid(logits)
                    eval_preds.extend((probs > 0.5).float().cpu().numpy())
                    eval_labels.extend(y_batch.numpy())
            
            # Calculate accuracies
            train_acc = balanced_accuracy_score(train_labels, train_preds)
            eval_acc = balanced_accuracy_score(eval_labels, eval_preds)
            
            # Update learning rate scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            if verbose:
                print(f"  Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | "
                      f"Train: {train_acc:.4f} | Eval: {eval_acc:.4f} | LR: {current_lr:.2e}")
            
            # Early stopping
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_weights = self.model.state_dict().copy()
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best weights
        if best_weights:
            self.model.load_state_dict(best_weights)
        
        self.is_trained = True
        
        # Evaluate
        results = self._evaluate_torch(X_eval_prep, y_eval_prep, "Evaluation")
        
        # Cleanup
        del train_loader, eval_loader
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    def _evaluate_torch(self, X, y, set_name):
        """Evaluate using torch tensors"""
        self.model.eval()
        all_preds, all_probs = [], []
        
        # Create dataloader for batch processing
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch).squeeze()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        bal_acc = balanced_accuracy_score(y, all_preds)
        try:
            auroc = roc_auc_score(y, all_probs)
        except ValueError:
            auroc = 0.5
        
        conf_matrix = confusion_matrix(y, all_preds)
        
        results = {
            'balanced_acc': bal_acc,
            'auroc': auroc,
            'confusion_matrix': conf_matrix,
            'predictions': np.array(all_preds)
        }
        
        print(f"\\n{self.name} - {set_name} Results:")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  Confusion Matrix: TN={conf_matrix[0,0]}, FP={conf_matrix[0,1]}, FN={conf_matrix[1,0]}, TP={conf_matrix[1,1]}")
        
        return results
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_tensor = torch.from_numpy(X).float()
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch).squeeze()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_tensor = torch.from_numpy(X).float()
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch).squeeze()
                probs = torch.sigmoid(logits)
                binary_probs = torch.stack([1-probs, probs], dim=1)
                probabilities.extend(binary_probs.cpu().numpy())
        
        return np.array(probabilities)