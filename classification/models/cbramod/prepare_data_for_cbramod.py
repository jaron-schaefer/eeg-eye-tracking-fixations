import numpy as np
from scipy.signal import resample
from data.data_configs import EEG_SAMPLE_FREQUENCY

def prepare_data_for_cbramod(X, y, subject_ids, stimulus_summary=None, orig_fs=EEG_SAMPLE_FREQUENCY):
    """
    Prepares EEG data for the CBraMod Foundation Model pipeline, ensuring all
    metadata arrays (labels, IDs, stimulus info) remain synchronized when
    bad trials are dropped.

    Processing Steps based on Wang et al. (2025):
    1. Resample to 200 Hz.
    2. Bad Sample Removal (> 100 µV).
    3. Normalization (1 unit = 100 µV).
    4. Patching (1-second segments).
    """
    
    # --- SAFETY CHECK: Input Synchronization ---
    # Convert inputs to numpy arrays to ensure boolean masking works
    y = np.array(y)
    subject_ids = np.array(subject_ids)
    if stimulus_summary is not None:
        stimulus_summary = np.array(stimulus_summary)

    # Verify all arrays have the same number of trials (length)
    if not (len(X) == len(y) == len(subject_ids)):
        raise ValueError(
            f"Length mismatch! X: {len(X)}, y: {len(y)}, ids: {len(subject_ids)}. "
            "All inputs must have the same number of trials."
        )
    
    if stimulus_summary is not None and len(stimulus_summary) != len(X):
         raise ValueError(f"Stimulus summary length {len(stimulus_summary)} does not match X {len(X)}")

    # --- STEP 1: Resample to 200 Hz ---
    # The model strictly requires 200 Hz input. We resample BEFORE filtering 
    # to catch any artifacts introduced by the resampling process itself.
    target_fs = 200
    if orig_fs != target_fs:
        print(f"Resampling {len(X)} trials from {orig_fs}Hz to {target_fs}Hz...")
        n_times = X.shape[2]
        new_n_times = int(n_times * target_fs / orig_fs)
        # Resample along the time axis (axis 2)
        X = resample(X, new_n_times, axis=2)
    else:
        print("Data is already at 200 Hz. Skipping resampling.")

    # --- STEP 2: Create Filter Mask ---
    # The paper removes samples where "any data point has an absolute amplitude exceeding 100 µV".
    # We create a boolean mask: True = Good Trial, False = Bad Trial.
    print(f"Checking {len(X)} trials for artifacts (>100µV or <-100µV)...")
    valid_mask = np.all((X <= 100) & (X >= -100), axis=(1, 2))

    # --- STEP 3: Apply Mask Synchronously ---
    # Filter X and ALL associated metadata arrays using the same mask.
    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]
    ids_filtered = subject_ids[valid_mask]
    
    # Filter stimulus_summary if it exists; otherwise keep it None
    if stimulus_summary is not None:
        stimulus_final = stimulus_summary[valid_mask]
    else:
        stimulus_final = None

    n_dropped = len(X) - len(X_filtered)
    print(f"Dropped {n_dropped} bad trials. Remaining: {len(X_filtered)}")

    if len(X_filtered) == 0:
        raise ValueError("All data was filtered out! Check your signal units (must be µV).")

    # --- STEP 4: Physical Scaling ---
    # "Normalize EEG by setting the unit to 100 µV".
    # This transforms the range roughly to [-1, 1].
    X_scaled = X_filtered / 100.0

    # --- STEP 5: Patching & Reshaping ---
    # The model requires input shape: (Batch, Channels, Patches, 200).
    # Each patch is exactly 1 second (200 samples).
    batch_size, n_channels, n_times = X_scaled.shape
    patch_length = 200 
    
    n_patches = n_times // patch_length
    
    if n_patches < 1:
        raise ValueError(f"Trials are too short! {n_times} samples is less than 200 (1s).")

    # Truncate any extra samples that don't fit into a full 1-second patch
    if n_times % patch_length != 0:
        print(f"Truncating {n_times % patch_length} extra timepoints per trial to fit patch size.")

    X_final = X_scaled[:, :, :n_patches * patch_length]
    X_final = X_final.reshape(batch_size, n_channels, n_patches, patch_length)

    print(f"Final Data Shape: {X_final.shape}")
    
    return X_final, y_filtered, ids_filtered, stimulus_final