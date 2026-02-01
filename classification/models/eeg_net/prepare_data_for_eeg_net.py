import numpy as np

def prepare_data_for_eeg_net(X, y, subject_ids):
    """
    Minimalist Data Prep for EEGNet.
    1. Per-Subject Z-Score Scaling.
    2. Reshape for Keras (Trials, Channels, Time, 1).
    """
    
    # Safety Check
    if not (len(X) == len(y) == len(subject_ids)):
        raise ValueError("Length mismatch in inputs!")
    
    # 1. Per-Subject Z-Score Normalization
    print("Applying Per-Subject Z-Score Normalization...")
    X_scaled = np.zeros_like(X)
    unique_subs = np.unique(subject_ids)
    
    for sub_id in unique_subs:
        # Get indices for this subject
        sub_indices = np.where(subject_ids == sub_id)[0]
        sub_data = X[sub_indices]
        
        # Calculate stats (mean/std) for this specific subject across all their trials/time
        sub_mean = np.mean(sub_data, axis=(0, 2), keepdims=True)
        sub_std = np.std(sub_data, axis=(0, 2), keepdims=True)
        
        # Apply scaling (adding epsilon to avoid div by zero)
        X_scaled[sub_indices] = (sub_data - sub_mean) / (sub_std + 1e-6)

    # 2. Reshape for Keras (Batch, Chans, Time, 1)
    # The '1' is the "channel/filter" dimension Keras Conv2D expects
    X_final = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], X_scaled.shape[2], 1)
    
    print(f"Final Data Shape: {X_final.shape}")
    
    return X_final, np.array(y)