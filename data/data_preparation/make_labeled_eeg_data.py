import os
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm

from data.data_configs import *


def load_fixations(subject: str) -> pd.DataFrame:
    events_path = os.path.join(LABELED_ET_DATA_DIR, f"{subject}_labeled_events.csv")
    df = pd.read_csv(events_path, dtype={"stimulus": str})
    fix = df[df["event_type"] == "Fixation"].sort_values(by="start_time")
    if fix["fixation_on_airplane"].isna().any():
        raise ValueError(f"NaN values in 'fixation_on_airplane' for {subject}")
    return fix


def load_eeg(subject: str, eeg_dir = EEG_DIR):
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
    channel_names = list(expected)
    return raw, data, times, channel_names


def epoch_fixations(data: np.ndarray, times: np.ndarray, fix_df: pd.DataFrame, pre_s: float, post_s: float):
    """
    Produce per-fixation EEG segments and labels.
    Returns only segments and labels; consumers use a common time axis
    relative to the fixation where 0 aligns to the event.
    """

    X_segments: list[np.ndarray] = []
    y: list[int] = []
    for _, row in fix_df.iterrows():
        fix_sec = row["eeg_time_start"] / 1000 #convert to seconds
        label = row["fixation_on_airplane"]
        i0 = max(0, np.searchsorted(times, fix_sec - pre_s))
        i1 = min(data.shape[1], np.searchsorted(times, fix_sec + post_s))
        X_segments.append(data[:, i0:i1])
        y.append(label)

    # print(X_segments)
    return X_segments, y


def filter_segments(segments: list[np.ndarray], labels: list[int], n_channels: int, expected_samples: int, subject: str, max_removed: int = 50,):
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


def save_training_npz(subject: str, pre_ms: int, post_ms: int, X: np.ndarray, y: np.ndarray, channel_names: list[str], sampling_rate: float, time_axis: np.ndarray):
    sub_dir = os.path.join(LABELED_EEG_DATA_DIR, subject)
    os.makedirs(sub_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(sub_dir, f"{pre_ms}_{post_ms}_training_data.npz"),
        X=X,
        y=y,
        time=time_axis,
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
    pre_s = pre_window_ms / 1000
    post_s = post_window_ms / 1000

    for subject in tqdm(subjects, desc="Processing subjects"):
        fix_df = load_fixations(subject)
        raw, data, times, channel_names = load_eeg(subject)

        segments, labels = epoch_fixations(data, times, fix_df, pre_s, post_s)
        fs = raw.info["sfreq"]
        expected_samples = int((pre_s + post_s) * fs)
        X, y, removed = filter_segments(segments, labels, n_channels=data.shape[0], expected_samples=expected_samples, subject=subject)
        if removed:
            print(f"Removed {removed} segments with wrong shape (expected {(data.shape[0], expected_samples)})")

        # Common per-segment time vector relative to event: [-pre_s, ..., +post_s)
        time_axis = (np.arange(expected_samples) / fs) - pre_s

        save_training_npz(subject, pre_window_ms, post_window_ms, X, y, channel_names, fs, time_axis)
        print(f"Saved {len(X)} segments for {subject} with {len(channel_names)} channels")


if __name__ == "__main__":
    make_training_data(subjects=SUBJECTS[0:30])
