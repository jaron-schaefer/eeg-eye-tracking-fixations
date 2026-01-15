import numpy as np
from data.data_configs import EEG_SAMPLE_FREQUENCY

def transform_et_time_to_eeg_time(et_timestamp, b):
    """
    Convert an eye-tracking (ET) timestamp into EEG-aligned time in milliseconds.
    
    Units:
    - ET timestamps (`et_timestamp`) are in **0.001 milliseconds units** (very high precision)
      e.g., ET records 1 ms as 1000 units.
    - Alignment parameter `b` is provided for each subject and is in the **same units as ET timestamps**.
      This represents the temporal discrepancy between EEG and ET recordings.
    
    Steps:
    1. Subtract `b` from the ET timestamp. This aligns the ET timeline to EEG timeline.
       Units are still in 0.001 ms.
    2. Divide by 1000 to convert the result into **milliseconds**.
       Resulting `EEG-aligned ET time` is now in standard milliseconds (ms).
    
    """
    et_timestamp = np.asarray(et_timestamp)  # ensures arrays are handled
    return (et_timestamp - b) / 1000.0

def transform_eeg_time_to_ms(time):
    """
    Convert EEG time (which is in seconds) into ms
    """
    time = np.asarray(time)
    return time * 1000.0



# not really needed because mne already has time
def transform_eeg_index_to_ms(sample_index, sfreq=EEG_SAMPLE_FREQUENCY):
    """
    Convert EEG sample indices to time in milliseconds.
    
    Units:
    - EEG data is recorded in **sample indices** (integers), not time.
      Each sample represents 1 / sfreq seconds.
      For example, if sfreq = 500 Hz, 1 sample = 0.002 seconds = 2 ms.
    - To get time in **milliseconds**:
        1. Divide the sample index by the sampling frequency → seconds
        2. Multiply by 1000 → milliseconds
    """
    sample_index = np.asarray(sample_index)
    return sample_index / sfreq * 1000.0
