import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from einops.layers.torch import Rearrange
from sklearn.metrics import (
    balanced_accuracy_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)

from classification.models.cbramod.cbramod import CBraMod

def check_pytorch_gpu():
    if torch.cuda.is_available():
        print(f"✅ PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Usage: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    else:
        print("❌ PyTorch is running on CPU! (Training will be very slow)")

# ==========================================
# 1. Data Loader Helper (Updated for 3 splits)
# ==========================================
def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    """ 
    Creates memory-efficient DataLoaders for Train, Val, and optional Test.
    """
    # 1. Train & Val (Mandatory)
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds   = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # 2. Test (Optional)
    test_loader = None
    if X_test is not None and y_test is not None:
        test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 3. Calculate Input Dim (Channels * Patches * Time)
    _, n_channels, n_patches, n_time = X_train.shape
    input_dim = n_channels * n_patches * n_time
    
    # 4. Cleanup to save RAM
    # Note: We delete the input arrays to free memory immediately
    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()
    
    return train_loader, val_loader, test_loader, input_dim

# ==========================================
# 2. Main Training Function (Comparable to EEGNet)
# ==========================================
def train_cbramod(X_train, y_train, 
                  X_val, y_val,             # MANDATORY: For Early Stopping
                  X_test=None, y_test=None, # OPTIONAL: For Final Report
                  lr=5e-4, 
                  dropout=0.1, 
                  hidden_dim=256,
                  epochs=30,
                  physical_batch_size=8,   
                  target_batch_size=64,    
                  weights_path='pretrained_weights.pth',
                  verbose=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_pytorch_gpu()
    
    # Calculate Accumulation Steps
    accum_steps = target_batch_size // physical_batch_size
    if verbose: 
        print(f"\n[Setup] CBraMod | LR: {lr} | Target Batch: {target_batch_size} (Accum: {accum_steps})")

    # --- 1. Setup Data & Model ---
    train_loader, val_loader, test_loader, input_dim = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, physical_batch_size
    )
    
    model = CBraMod().to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except FileNotFoundError:
        print(f"Warning: Weights not found at {weights_path}")

    # Dynamic Classification Head
    model.proj_out = nn.Identity()
    model.classifier = nn.Sequential(
        Rearrange('b c s p -> b (c s p)'),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1) 
    ).to(device)
    
    # --- 2. Optimization Setup ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda') 

    best_val_score = 0.0
    best_weights = None
    patience = 0
    patience_limit = 20 # Same as EEGNet
    
    # --- 3. Training Loop ---
    if verbose: print("[Status] Training (Monitoring X_val for best weights)...")
    
    for epoch in range(epochs):
        # A. Train
        model.train()
        train_loss = 0.0
        optimizer.zero_grad() 
        
        for i, (X_b, y_b) in enumerate(train_loader):
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model.classifier(model(X_b))
                loss = criterion(logits.squeeze(), y_b)
                loss = loss / accum_steps 
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * accum_steps
            
        # B. Validate (On X_val)
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model.classifier(model(X_b))
                
                probs = torch.sigmoid(logits.squeeze())
                preds.extend((probs > 0.5).float().cpu().numpy())
                labels.extend(y_b.numpy())
        
        # C. Score (Use Balanced Accuracy for Early Stopping)
        val_acc = balanced_accuracy_score(labels, preds)
        
        if verbose:
            print(f"Epoch {epoch+1:<3} | Loss: {train_loss/len(train_loader):.4f} | Val Bal Acc: {val_acc:.4f}")

        # D. Early Stopping Logic (Save Best Weights)
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_weights = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                if verbose: print("Early stopping triggered.")
                break

    # --- 4. Final Evaluation Logic ---
    # Restore Best Weights (Simulating 'restore_best_weights=True')
    if best_weights:
        model.load_state_dict(best_weights)
        
    # Determine which set to evaluate on
    if test_loader is not None:
        if verbose: print("\n[Status] Best weights restored. Evaluating on TEST set...")
        target_loader = test_loader
    else:
        if verbose: print("\n[Status] Best weights restored. Evaluating on VALIDATION set (Optimization mode)...")
        target_loader = val_loader

    # Run Final Evaluation
    model.eval()
    final_preds, final_labels, final_probs = [], [], []
    
    with torch.no_grad():
        for X_b, y_b in target_loader:
            X_b = X_b.to(device)
            with torch.amp.autocast('cuda'):
                logits = model.classifier(model(X_b))
            
            batch_probs = torch.sigmoid(logits.squeeze())
            final_probs.extend(batch_probs.cpu().numpy())
            final_preds.extend((batch_probs > 0.5).float().cpu().numpy())
            final_labels.extend(y_b.numpy())

    # --- 5. Metrics Calculation ---
    bal_acc = balanced_accuracy_score(final_labels, final_preds)
    
    try:
        auroc = roc_auc_score(final_labels, final_probs)
    except ValueError:
        auroc = 0.5

    results = {
        'score': bal_acc, # Main metric for optimization
        'balanced_acc': bal_acc, # Explicit name
        'confusion_matrix': confusion_matrix(final_labels, final_preds),
        'report': classification_report(final_labels, final_preds, target_names=['Non-Target', 'Target'], zero_division=0),
        'auroc': auroc
    }

    if verbose:
        print("\n--- Final Results (Best Epoch) ---")
        print(results['report'])
        print(f"Balanced Acc: {bal_acc:.4f} | AUROC: {auroc:.4f}")

    # Cleanup
    del model, optimizer, scaler, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return results