"""
Utilities for loading and processing EEG data from NPZ files with proper channel mapping.
"""

import numpy as np
import warnings
from typing import List, Optional, Sequence, Tuple, Union
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils as np_utils
from pathlib import Path
import os
from data.data_configs import EEG_CHANNEL_NAMES, DEFAULT_EEGNET_CHANNELS, LABELED_EEG_DATA_DIR


class DataLoader:

    @classmethod
    def load_subject_labeled_eeg(
        cls, 
        subject: str, 
        channels: List[str] = EEG_CHANNEL_NAMES,
        pre_window_ms: int = 500, 
        post_window_ms: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
        
        path = LABELED_EEG_DATA_DIR / subject / f"{pre_window_ms}_{post_window_ms}_training_data.npz"
        data = np.load(str(path))

        X = data["X"] # (fixations, channels, samples)
        y = data["y"]
        channel_names = data["channel_names"].tolist()

        missing = [ch for ch in channels if ch not in channel_names]
        if missing:
            raise ValueError(f"The following channels are not in the data: {missing}")

        indices = [channel_names.index(ch) for ch in channels]
        X = X[:, indices, :]
        channel_names = [channel_names[i] for i in indices]

        metadata = {
            "sampling_rate": float(data["sampling_rate"]),
            "pre_window_ms": int(data["pre_window_ms"]),
            "post_window_ms": int(data["post_window_ms"]),
            "subject": subject,
            "time": data["time"],
        }

        assert channels == channel_names
        return X, y, channel_names, metadata

    
    @classmethod
    def load_labeled_eeg_data(
        cls,
        subjects: Union[str, Sequence[str]],
        channels: Optional[Union[str, Sequence[str]]] = EEG_CHANNEL_NAMES,
        pre_window_ms: int = 500,
        post_window_ms: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Normalize single subject or channel to list
        subject_list = [subjects] if isinstance(subjects, str) else list(subjects)
        channel_list = [channels] if isinstance(channels, str) else list(channels)

        X_blocks: List[np.ndarray] = []
        y_blocks: List[np.ndarray] = []
        time_axis: Optional[np.ndarray] = None
        selected_names_ref: Optional[List[str]] = None

        for subj in subject_list:
            X, y, selected_names_curr, metadata = cls.load_subject_labeled_eeg(
                subj, channels=channel_list, pre_window_ms=pre_window_ms, post_window_ms=post_window_ms
            )

            # Ensure consistent channel order across subjects
            if selected_names_ref is None:
                selected_names_ref = selected_names_curr
            elif selected_names_curr != selected_names_ref:
                raise ValueError(
                    f"Channel order mismatch for subject {subj}. "
                    f"Expected {selected_names_ref}, got {selected_names_curr}"
                )

            X_blocks.append(X)
            y_blocks.append(y)

            # Capture and check consistent time axis
            if time_axis is None:
                if metadata["time"] is None:
                    raise KeyError("'time' axis missing from NPZ.")
                time_axis = metadata["time"]
            elif not np.allclose(time_axis, metadata["time"]):
                raise ValueError("Time axis mismatch across subjects.")

        X_out = np.concatenate(X_blocks, axis=0)
        y_out = np.concatenate(y_blocks, axis=0)

        return X_out, y_out, time_axis

    def load_segment_time_axis(
        self, subject: str, pre_window_ms: int = 500, post_window_ms: int = 500
    ) -> np.ndarray:
        """
        Load the common per-segment time vector from the subject NPZ.

        Returns a 1D array aligned to the segment window, typically
        spanning [-pre_s, ..., +post_s) with sampling based on the stored rate.
        """
        path = LABELED_EEG_DATA_DIR / subject / f"{pre_window_ms}_{post_window_ms}_training_data.npz"
        data = np.load(str(path))
        if "time" not in data.files:
            raise KeyError("'time' not found in NPZ for subject. Regenerate data to include time axis.")
        return data["time"]




# Backward compatible module-level wrappers (used by older code paths)
def load_training_data(subject: str, pre_window_ms: int = 500, post_window_ms: int = 500):
    dl = DataLoader()
    return dl._load_subject_labeled_eeg(subject, pre_window_ms, post_window_ms)


def select_channels_by_name(X, available_channels, desired_channels):
    dl = DataLoader()
    return dl.select_channels_by_name(X, available_channels, desired_channels)





def prepare_data_for_eegnet(X, y, test_size=0.3, val_size=0.5, random_state=42):
    
    # Add kernel dimension for EEGNet (channels_last format)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, stratify=y, random_state=random_state
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, shuffle=True, stratify=y_temp, random_state=random_state
    )
    
    # Convert labels to categorical (one-hot encoding)
    n_classes = len(np.unique(y))
    y_train = np_utils.to_categorical(y_train.astype(int), num_classes=n_classes)
    y_val = np_utils.to_categorical(y_val.astype(int), num_classes=n_classes)
    y_test = np_utils.to_categorical(y_test.astype(int), num_classes=n_classes)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_and_prepare_eegnet_data(subject, desired_channels=None, **kwargs):
    
    if desired_channels is None:
        desired_channels = DEFAULT_EEGNET_CHANNELS
    
    # Load raw data via DataLoader for backward compatibility
    dl = DataLoader()
    X, y, channel_names, _metadata = dl._load_subject_labeled_eeg(subject, 500, 500)
    
    # Select desired channels
    X_selected, _idx, selected_channels = dl.select_channels_by_name(
        X, channel_names, desired_channels
    )
    
    # Prepare for EEGNet
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_eegnet(
        X_selected, y, **kwargs
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, selected_channels

def get_channel_info(subject):
    """
    Quick function to check what channels are available for a subject.
    """
    try:
        dl = DataLoader()
        X, y, channel_names, metadata = dl._load_subject_labeled_eeg(subject, 500, 500)
        print(f"\nSubject {subject} channel info:")
        print(f"Available channels ({len(channel_names)}): {channel_names}")
        print(f"Data shape: {X.shape}")
        print(f"Available in DEFAULT_EEGNET_CHANNELS: {[ch for ch in DEFAULT_EEGNET_CHANNELS if ch in channel_names]}")
        print(f"Missing from DEFAULT_EEGNET_CHANNELS: {[ch for ch in DEFAULT_EEGNET_CHANNELS if ch not in channel_names]}")
        return channel_names
    except FileNotFoundError:
        print(f"No training data found for {subject}")
        return None