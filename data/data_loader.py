from collections import Counter
import os
import warnings
import mne
import numpy as np
from pathlib import Path
import pandas as pd
import gc


from typing import List, Optional, Sequence, Tuple, Union
from sklearn.model_selection import train_test_split
from data.data_configs import *


class DataLoader:
    
    # Class attribute for data path - can be changed when needed
    data_path = Path(DATA_ROOT)
    
    def __init__(self, data_path=None):
        """Initialize DataLoader with data path.
        
        Args:
            data_path: Path to the data directory. If None, uses DATA_ROOT from configs.
        """
        if data_path is not None:
            # Update class attribute if instance is created with custom path
            DataLoader.data_path = Path(data_path)
    
    @classmethod
    def _get_data_path(cls, *path_components):
        """Get full path by joining data_path with relative path components."""
        return cls.data_path.joinpath(*path_components)

    @classmethod
    def load_eeg(cls, subject: str):
        """Load EEG data for a subject from raw_data/EEG_clean data directory."""
        path = cls._get_data_path("raw_data", "EEG_clean data", f"{subject}.edf")
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

        actual = [ch for ch in raw.ch_names if ch != "Status"]
        expected = EEG_CHANNEL_NAMES
        missing = [ch for ch in expected if ch not in actual]
        extra = [ch for ch in actual if ch not in expected]
        if missing or extra:
            raise ValueError(
                f"Channel set mismatch for {subject}!\nMissing: {missing}\nExtra: {extra}\n"
                f"Expected ({len(expected)}): {expected}\nActual ({len(actual)}): {actual}"
            )

        picks = [raw.ch_names.index(ch) for ch in expected]
        data, times = raw.get_data(picks=picks, return_times=True)
        data = data * 1e6  # V -> μV conversion
        return raw, data, times, list(expected)

    @classmethod
    def load_labeled_et_data(
        cls,
        subjects: Union[str, Sequence[str]]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Load labeled eyetracking data for one or more subjects from labeled_et_data directory.
        """
        subj_list = [subjects] if isinstance(subjects, str) else list(subjects)
        df_list = []
        stimulus_summary = None
        for subj in subj_list:
            et_path = cls._get_data_path("labeled_et_data", f"{subj}_labeled_events.csv")
            df = pd.read_csv(et_path)
            df = df[df["event_type"] == "Fixation"]
            df['subject'] = subj
            if 'stimulus_summary' in df.columns:
                if stimulus_summary is None:
                    stimulus_summary = df['stimulus_summary']
            df_list.append(df)
        big_df = pd.concat(df_list, ignore_index=True)
        return big_df, stimulus_summary

    @classmethod
    def load_features(
        cls,
        subjects: Union[str, Sequence[str]],
        mode: str = "advanced", 
        channels: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Optional[pd.Series]]:
        """Load EEG features with optional channel filtering.
        """

        subj_list = [subjects] if isinstance(subjects, str) else list(subjects)
        X_blocks = []
        y_blocks = []
        subject_ids = []
        feature_names = None
        stimulus_summary = None
        
        for subj in subj_list:
            # Use relative path from data_path
            feat_path = cls._get_data_path("features", mode, subj, "features.csv")
            if not feat_path.exists():
                print(f"Warning: Feature file not found for {subj}: {feat_path}")
                continue
            df = pd.read_csv(feat_path)
            
            # Filter features by channels if specified
            if channels is not None:
                # Select columns where feature name starts with channel name + '_'
                selected_cols = []
                for col in df.columns:
                    # Skip non-feature columns
                    if col in ['label', 'stimulus_summary']:
                        continue
                    # Check if column starts with any of the specified channels
                    for ch in channels:
                        if col.startswith(f'{ch}_'):
                            selected_cols.append(col)
                            break
                
                # Always include label and stimulus_summary if present
                for meta_col in ['label', 'stimulus_summary']:
                    if meta_col in df.columns and meta_col not in selected_cols:
                        selected_cols.append(meta_col)
                        
                df = df[selected_cols]
                print(f"Filtered to {len(selected_cols)-1} features for channels {channels} (subject {subj})")
            
            # Handle stimulus summary
            if 'stimulus_summary' in df.columns:
                if stimulus_summary is None:
                    stimulus_summary = df['stimulus_summary']
                df = df.drop(columns=['stimulus_summary'])
            
            # Extract features and labels
            if 'label' in df.columns:
                X = df.drop(columns=["label"]).values
                y = df["label"].values
                current_feature_names = list(df.drop(columns=["label"]).columns)
            else:
                X = df.values
                y = None
                current_feature_names = list(df.columns)

            if feature_names is None:
                feature_names = current_feature_names
            elif feature_names != current_feature_names:
                print(f"Warning: Feature names mismatch for subject {subj}")
                
            X_blocks.append(X)
            if y is not None:
                y_blocks.append(y)
            subject_ids.extend([subj] * len(X))
        
        if not X_blocks:
            raise ValueError(f"No features loaded for subjects: {subj_list}")
            
        X_out = np.vstack(X_blocks)
        y_out = np.concatenate(y_blocks) if y_blocks else None
        return X_out, y_out, feature_names, subject_ids, stimulus_summary
    
    
    @classmethod
    def load_et_features(
        cls,
        subjects: Union[str, Sequence[str]],
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Optional[pd.Series]]:
        """Load eye tracking features."""
        subj_list = [subjects] if isinstance(subjects, str) else list(subjects)
        X_blocks = []
        y_blocks = []
        subject_ids = []
        feature_names = None
        stimulus_summary = None
        for subj in subj_list:
            # Use relative path from data_path
            feat_path = cls._get_data_path("features", "et_features", f"{subj}_features.csv")
            if not feat_path.exists():
                print(f"Warning: Feature file not found for {subj}: {feat_path}")
                continue
            df = pd.read_csv(feat_path)
            # Define columns to exclude from features (metadata and label)
            exclude_cols = ['subject', 'stimulus', 'fixation_on_airplane', 'stimulus_summary']
            if 'stimulus_summary' in df.columns:
                if stimulus_summary is None:
                    stimulus_summary = df['stimulus_summary']
            if 'fixation_on_airplane' in df.columns:
                y = df['fixation_on_airplane'].values
            else:
                raise ValueError(f"No 'fixation_on_airplane' column found in {feat_path}")
            # Get feature columns (exclude metadata and label)
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            X = df[feature_cols].values
            # Store feature names from first subject
            if feature_names is None:
                feature_names = feature_cols
            elif feature_cols != feature_names:
                print(f"Warning: Feature mismatch for {subj}. Expected {len(feature_names)}, got {len(feature_cols)}")
            X_blocks.append(X)
            y_blocks.append(y)
            subject_ids.extend([subj] * len(X))
        if not X_blocks:
            raise ValueError(f"No features loaded for subjects: {subj_list}")
        X_out = np.vstack(X_blocks)
        y_out = np.concatenate(y_blocks)
        return X_out, y_out, feature_names, subject_ids, stimulus_summary
    

    @classmethod
    def load_subject_labeled_eeg(
        cls, 
        subject: str,
        channels: List[str] = EEG_CHANNEL_NAMES,
        pre_window_ms: int = PRE_WINDOW_MS, 
        post_window_ms: int = POST_WINDOW_MS
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, dict]:
        """
        Loads subject EEG data and returns X, y, channel_names, stimulus_summary, metadata
        """
        path = cls._get_data_path("labeled_eeg_data", subject, f"{pre_window_ms}_{post_window_ms}_training_data.npz")
        data = np.load(str(path))

        X = data["X"] # (fixations, channels, samples)
        y = data["y"]
        channel_names = data["channel_names"].tolist()
        stimulus_summary = data["stimulus_summary"]

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
        return X, y, channel_names, stimulus_summary, metadata

    
    # ============================================================================
    # COLAB-SPECIFIC DATA LOADING FUNCTIONS
    # ============================================================================
    @classmethod
    def load_labeled_eeg_data(cls, subjects, channels,
                                epoch_start_ms=EPOCH_START_MS,
                                epoch_end_ms=EPOCH_END_MS,
                                pre_ms=PRE_WINDOW_MS,
                                post_ms=POST_WINDOW_MS,
                                fs=EEG_SAMPLE_FREQUENCY):
        """
        Load EEG data AND stimulus summary together.
        """
        print("\n" + "="*70)
        print("LOADING EEG DATA WITH STIMULUS SUMMARY")
        print("="*70)
        
        all_X, all_y, all_subjects, all_stim = [], [], [], []
        
        # Calculate epoch indices
        total_time_points = int((pre_ms + post_ms) / 1000 * fs)
        fixation_idx = int(pre_ms / 1000 * fs)
        epoch_start_idx = fixation_idx + int(epoch_start_ms / 1000 * fs)
        epoch_end_idx = fixation_idx + int(epoch_end_ms / 1000 * fs)
        
        print(f"\nEpoch settings:")
        print(f"  Window: [-{pre_ms}, {post_ms}] ms")
        print(f"  Epoch: [{epoch_start_ms}, {epoch_end_ms}] ms")
        print(f"  Sampling rate: {fs} Hz")
        
        # Load data from each subject
        print(f"\nLoading from {len(subjects)} subjects...")
        for subject in subjects:
            file_path = cls._get_data_path("labeled_eeg_data", subject, f"{pre_ms}_{post_ms}_training_data.npz")
            
            if not file_path.exists():
                print(f" {subject}: File not found")
                continue
            
            # Load all data
            with np.load(file_path, mmap_mode='r') as data:
                ch_names = data['channel_names']
                
                # Find channel indices
                ch_indices = [list(ch_names).index(ch) for ch in channels if ch in ch_names]
                
                if len(ch_indices) == 0:
                    print(f"  ⚠ {subject}: No matching channels")
                    continue
                
                # Load EEG
                X_selected = data['X'][:, ch_indices, epoch_start_idx:epoch_end_idx].copy().astype(np.float32)
                y = data['y'][:]
                
                # Load stimulus
                if 'stimulus_summary' in data:
                    stim = data['stimulus_summary'][:].copy()
                elif 'stimulus' in data:
                    stim = data['stimulus'][:].copy()
                else:
                    print(f"  ⚠ {subject}: No stimulus data, using zeros")
                    stim = np.zeros(len(y))
            
            all_X.append(X_selected)
            all_y.append(y)
            all_stim.append(stim)
            all_subjects.extend([subject] * len(y))
            
            del X_selected, stim
            gc.collect()
        
        # Combine all subjects
        X_combined = np.concatenate(all_X, axis=0).astype(np.float32)
        y_combined = np.concatenate(all_y, axis=0)
        stimulus_combined = np.concatenate(all_stim, axis=0)
        del all_X, all_y, all_stim
        gc.collect()
        
        # Print summary
        print(f"\n✓ Data loaded:")
        print(f"  EEG: {X_combined.shape}")
        print(f"  Labels: {y_combined.shape}")
        print(f"  Stimulus: {stimulus_combined.shape}")
        print(f"  Subjects: {len(set(all_subjects))}")
        
        # Print label distribution
        label_counts = Counter(y_combined)
        total_samples = len(y_combined)
        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            proportion = count / total_samples
            print(f"  Label {label}: {count} samples ({proportion:.2%})")
        
        return X_combined, y_combined, np.array(all_subjects), stimulus_combined
