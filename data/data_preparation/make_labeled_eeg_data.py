import os
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm

from data.data_configs import *
from data.data_utils import transform_eeg_time_to_ms


def load_fixations(subject: str) -> pd.DataFrame:
    """Load labeled fixation events for a subject from CSV file."""
    events_path = os.path.join(LABELED_ET_DATA_DIR, f"{subject}_labeled_events.csv")
    df = pd.read_csv(events_path, dtype={"stimulus": str})
    fix = df[df["event_type"] == "Fixation"].sort_values(by="eeg_time_start")
    if fix["fixation_on_airplane"].isna().any():
        raise ValueError(f"NaN values in 'fixation_on_airplane' for {subject}")
    return fix


def load_eeg(subject: str, eeg_dir = EEG_DIR):
    """Load and validate EEG data from EDF file with proper channel ordering."""
    path = os.path.join(eeg_dir, f"{subject}.edf")
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    # Exclude status channel and validate set equality
    actual = [ch for ch in raw.ch_names if ch != "Status"]
    expected = EEG_CHANNEL_NAMES
    missing = [ch for ch in expected if ch not in actual]
    extra = [ch for ch in actual if ch not in expected]
    if missing or extra:
        raise ValueError(
            f"Channel set mismatch for {subject}!\nMissing: {missing}\nExtra: {extra}\n"
            f"Expected ({len(expected)}): {expected}\nActual ({len(actual)}): {actual}"
        )

    # Reorder to canonical expected order to prevent mix-ups
    picks = [raw.ch_names.index(ch) for ch in expected]
    data, times = raw.get_data(picks=picks, return_times=True)
    
    # Convert from Volts to microvolts for better neural network training
    # MNE returns data in Volts (V), but values around 1e-05 V are too small for stable NN training
    # Scaling by 1e6 converts to microvolts (μV) and puts data in optimal range for NNs
    data = data * 1e6  # V -> μV conversion
    
    channel_names = list(expected)
    return raw, data, times, channel_names


def epoch_fixations(data: np.ndarray, times: np.ndarray, fix_df: pd.DataFrame, pre_ms: int, post_ms: int):
    """Extract EEG segments around each fixation event with corresponding labels."""

    # Convert EEG times from seconds to milliseconds for integer-based calculations
    times_ms = transform_eeg_time_to_ms(times)
    
    X_segments: list[np.ndarray] = []
    y: list[int] = []
    for _, row in fix_df.iterrows():
        # eeg_time_start is already in milliseconds from transform_et_time_to_eeg_time()
        fix_ms = row["eeg_time_start"]  # already in milliseconds (float)
        label = row["fixation_on_airplane"]
        # Find indices using millisecond precision
        i0 = max(0, np.searchsorted(times_ms, fix_ms - pre_ms))
        i1 = min(data.shape[1], np.searchsorted(times_ms, fix_ms + post_ms))
        X_segments.append(data[:, i0:i1])
        y.append(label)

    return X_segments, y


def filter_segments(segments: list[np.ndarray], labels: list[int], n_channels: int, expected_samples: int, subject: str, max_removed: int = 50,):
    """Remove segments that don't match the expected shape."""
    keep = []
    for i, seg in enumerate(segments):
        if seg.shape != (n_channels, expected_samples):
            continue
        keep.append(i)

    removed = len(segments) - len(keep)

    if removed >= max_removed:
        raise ValueError(f"Too many segments removed: {removed} >= {max_removed} for subject:{subject}")
    if removed:
        segments = [segments[i] for i in keep]
        labels = [labels[i] for i in keep]
    return np.array(segments), np.array(labels), removed


def save_training_npz(subject: str, pre_ms: int, post_ms: int, X: np.ndarray, y: np.ndarray, channel_names: list[str], sampling_rate: float, time_axis_ms: np.ndarray):
    """Save processed EEG training data to compressed NPZ file."""
    sub_dir = os.path.join(LABELED_EEG_DATA_DIR, subject)
    os.makedirs(sub_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(sub_dir, f"{pre_ms}_{post_ms}_training_data.npz"),
        X=X,
        y=y,
        time=time_axis_ms,
        channel_names=channel_names,
        sampling_rate=sampling_rate,
        pre_window_ms=pre_ms,
        post_window_ms=post_ms,
    )


def make_training_data(
    pre_window_ms: int = 500,
    post_window_ms: int = 500,
    subjects: list[str] = SUBJECTS
):
    """Process all subjects to create labeled EEG training datasets."""
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        fix_df = load_fixations(subject)
        raw, data, times, channel_names = load_eeg(subject)

        segments, labels = epoch_fixations(data, times, fix_df, pre_window_ms, post_window_ms)
        fs = raw.info["sfreq"]
        expected_samples = int((pre_window_ms + post_window_ms) * fs / 1000)  # convert ms to samples
        X, y, removed = filter_segments(segments, labels, n_channels=data.shape[0], expected_samples=expected_samples, subject=subject)
        if removed:
            print(f"Removed {removed} segments with wrong shape (expected {(data.shape[0], expected_samples)})")

        # Create standardized time axis for all epochs in milliseconds
        # Note: We can't use the original 'times' variable because it contains absolute timestamps
        # Instead, we need relative timestamps centered on the event (fixation onset)
        # This creates: [-pre_window_ms, -pre_window_ms+dt, ..., -dt, 0, +dt, ..., +post_window_ms) where 0 = fixation onset
        time_axis_ms = np.arange(expected_samples) * (1000.0 / fs) - pre_window_ms  # time axis in milliseconds

        save_training_npz(subject, pre_window_ms, post_window_ms, X, y, channel_names, fs, time_axis_ms)
        print(f"Saved {len(X)} segments for {subject} with {len(channel_names)} channels")


if __name__ == "__main__":
    make_training_data(subjects=SUBJECTS[14:15])
