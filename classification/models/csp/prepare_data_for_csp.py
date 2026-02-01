import numpy as np
import mne

from data.data_configs import EEG_SAMPLE_FREQUENCY

def prepare_data_for_csp(X, y, fs=EEG_SAMPLE_FREQUENCY):
    """
    Standard CSP Prep (Bandpass only).
    Does NOT require subject_ids.
    """
    print(f"--- Standard CSP Data Prep (Fs={fs}Hz) ---")
    
    # 1. MNE Conversion (for robust filtering)
    # We create a dummy info object just to use MNE's filter function
    n_trials, n_chans, n_times = X.shape
    info = mne.create_info(ch_names=n_chans, sfreq=fs, ch_types=['eeg']*n_chans)
    epochs = mne.EpochsArray(X, info, verbose=False)
    
    # 2. Bandpass Filter (0.5 - 40 Hz)
    # This removes slow drift (<0.5) and high freq noise (>40)
    print("Applying Bandpass Filter (0.5 - 40 Hz)...")
    epochs.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    
    # 3. Extract & Squeeze
    # Get the data back as a numpy array
    X_clean = epochs.get_data()
    
    # If dimensions are (Trials, Chans, Time, 1), squeeze to (Trials, Chans, Time)
    if X_clean.ndim == 4:
        X_clean = X_clean.squeeze(-1)

    print(f"Data Ready. Shape: {X_clean.shape}")
    
    return X_clean, np.array(y)