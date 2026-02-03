# =============================================================================
# NOTE: EEG data is stored in volts in the raw files, but when loaded using
# DataLoader (load_eeg or load_subject_labeled_eeg), it is converted to microvolts (μV).
# All processing and feature extraction in this script assumes data is in microvolts.
# =============================================================================
import os
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm

from data.data_configs import *
from data.data_loader import DataLoader
from data.data_utils import transform_eeg_time_to_ms


def load_fixations(subject: str) -> pd.DataFrame:
    events_path = os.path.join(SELECTED_ET_DATA_DIR, f"{subject}_labeled_events.csv")
    df = pd.read_csv(events_path, dtype={"stimulus": str})
    fix = df[df["event_type"] == "Fixation"].sort_values(by="eeg_time_start")
    if fix["fixation_on_airplane"].isna().any():
        raise ValueError(f"NaN values in 'fixation_on_airplane' for {subject}")
    return fix

def epoch_fixations(data: np.ndarray, times: np.ndarray, fix_df: pd.DataFrame, pre_ms: int, post_ms: int):
    times_ms = transform_eeg_time_to_ms(times)
    X_segments: list[np.ndarray] = []
    y: list[int] = []
    stimulus_summary: list[str] = []
    for _, row in fix_df.iterrows():
        fix_ms = row["eeg_time_start"]
        label = row["fixation_on_airplane"]
        i0 = max(0, np.searchsorted(times_ms, fix_ms - pre_ms))
        i1 = min(data.shape[1], np.searchsorted(times_ms, fix_ms + post_ms))
        X_segments.append(data[:, i0:i1])
        y.append(label)
        stimulus_summary.append(row["stimulus_summary"])
    return X_segments, y, stimulus_summary


def filter_segments(segments: list[np.ndarray], labels: list[int], stimulus_summary: list[str], n_channels: int, expected_samples: int, subject: str, max_removed: int = 50):
    keep = [i for i, seg in enumerate(segments) if seg.shape == (n_channels, expected_samples)]
    removed = len(segments) - len(keep)
    if removed >= max_removed:
        raise ValueError(f"Too many segments removed: {removed} >= {max_removed} for subject:{subject}")
    if removed:
        segments = [segments[i] for i in keep]
        labels = [labels[i] for i in keep]
        stimulus_summary = [stimulus_summary[i] for i in keep]
    return np.array(segments), np.array(labels), np.array(stimulus_summary), removed

def baseline_correct(segments: np.ndarray, fs: float, pre_ms: int, baseline_start_ms: int = -300, baseline_end_ms: int = 0):
    start_idx = int((baseline_start_ms + pre_ms) * fs / 1000)
    end_idx = int((baseline_end_ms + pre_ms) * fs / 1000)
    corrected = np.empty_like(segments)
    for i, seg in enumerate(segments):
        baseline = seg[:, start_idx:end_idx].mean(axis=1, keepdims=True)
        corrected[i] = seg - baseline
    return corrected


def save_training_npz(subject: str, pre_ms: int, post_ms: int,
                       X: np.ndarray, y: np.ndarray, stimulus_summary: np.ndarray, 
                       channel_names: list[str], sampling_rate: float, time_axis_ms: np.ndarray):
    sub_dir = os.path.join(LABELED_EEG_DATA_DIR, subject)
    os.makedirs(sub_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(sub_dir, f"{pre_ms}_{post_ms}_training_data.npz"),
        X=X,
        y=y,
        stimulus_summary=stimulus_summary,
        time=time_axis_ms,
        channel_names=channel_names,
        sampling_rate=sampling_rate,
        pre_window_ms=pre_ms,
        post_window_ms=post_ms,
    )


def label_eeg_data(subjects: list[str] = SUBJECTS, pre_window_ms: int = PRE_WINDOW_MS, post_window_ms: int = POST_WINDOW_MS):
    for subject in tqdm(subjects, desc="Processing subjects"):
        fix_df = load_fixations(subject)
        raw, data, times, channel_names = DataLoader.load_eeg(subject)

        segments, labels, stimulus_summary = epoch_fixations(data, times, fix_df, pre_window_ms, post_window_ms)
        fs = raw.info["sfreq"]
        expected_samples = int((pre_window_ms + post_window_ms) * fs / 1000)
        X, y, stimulus_summary, removed = filter_segments(segments, labels, stimulus_summary, n_channels=data.shape[0], expected_samples=expected_samples, subject=subject)
        if removed:
            print(f"Removed {removed} segments with wrong shape (expected {(data.shape[0], expected_samples)})")

        # Baseline correction
        X = baseline_correct(X, fs, pre_window_ms)

        # Time axis relative to fixation
        time_axis_ms = np.arange(expected_samples) * (1000.0 / fs) - pre_window_ms
        save_training_npz(subject, pre_window_ms, post_window_ms, X, y, stimulus_summary, channel_names, fs, time_axis_ms)
        print(f"Saved {len(X)} segments for {subject} with {len(channel_names)} channels")
