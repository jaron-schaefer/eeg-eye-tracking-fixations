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

    @classmethod
    def load_eeg(cls, subject: str, eeg_dir=EEG_DIR):
        path = os.path.join(eeg_dir, f"{subject}.edf")
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

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
        subjects: Union[str, Sequence[str]],
        et_data_dir: Path = LABELED_ET_DATA_DIR
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Load labeled eyetracking data for one or more subjects from LABELED_ET_DATA_DIR.
        Returns:
            df: Single DataFrame with all data and a 'subject' column
            stimulus_summary: pd.Series if present, else None
        """
        subj_list = [subjects] if isinstance(subjects, str) else list(subjects)
        df_list = []
        stimulus_summary = None
        for subj in subj_list:
            et_path = Path(et_data_dir) / f"{subj}_labeled_events.csv"
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
        channels: List[str] = EEG_CHANNEL_NAMES
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Optional[pd.Series]]:

        subj_list = [subjects] if isinstance(subjects, str) else list(subjects)
        X_blocks = []
        y_blocks = []
        subject_ids = []
        feature_names = None
        stimulus_summary = None
        for subj in subj_list:
            feat_path = Path(FEATURES_DIR) / mode / subj / "features.csv"
            df = pd.read_csv(feat_path)
            if channels is not None:
                # Select all columns where the channel name appears anywhere in the feature name (e.g., 'P7' in 'min_P7' or 'P7_min')
                selected_cols = [col for col in df.columns if any((ch in col or col in ch) for ch in channels)]
                # Always include label if present
                if 'label' in df.columns and 'label' not in selected_cols:
                    selected_cols.append('label')
                df = df[selected_cols]
            if 'stimulus_summary' in df.columns:
                if stimulus_summary is None:
                    stimulus_summary = df['stimulus_summary']
            if 'label' in df.columns:
                X = df.drop(columns=["label"]).values
                y = df["label"].values
                current_feature_names = list(df.drop(columns=["label"]).columns)

            if feature_names is None:
                feature_names = current_feature_names
            X_blocks.append(X)
            if y is not None:
                y_blocks.append(y)
            subject_ids.extend([subj] * len(X))
        X_out = np.vstack(X_blocks)
        y_out = np.concatenate(y_blocks) if y_blocks else None
        return X_out, y_out, feature_names, subject_ids, stimulus_summary
    
    
    @classmethod
    def load_et_features(
        cls,
        subjects: Union[str, Sequence[str]],
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Optional[pd.Series]]:

        subj_list = [subjects] if isinstance(subjects, str) else list(subjects)
        X_blocks = []
        y_blocks = []
        subject_ids = []
        feature_names = None
        stimulus_summary = None
        for subj in subj_list:
            feat_path = Path(FEATURES_DIR) / "et_features" / f"{subj}_features.csv"
            if not feat_path.exists():
                print(f"Warning: Feature file not found for {subj}: {feat_path}")
                continue
            df = pd.read_csv(feat_path)
            # Define columns to exclude from features (metadata and label)
            exclude_cols = ['subject', 'stimulus', 'fixation_on_airplane']
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
        labeled_eeg_data_dir: Path = LABELED_EEG_DATA_DIR,
        channels: List[str] = EEG_CHANNEL_NAMES,
        pre_window_ms: int = PRE_WINDOW_MS, 
        post_window_ms: int = POST_WINDOW_MS
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, dict]:
        """
        Loads subject EEG data and returns X, y, channel_names, stimulus_summary, metadata
        """
        path = labeled_eeg_data_dir / subject / f"{pre_window_ms}_{post_window_ms}_training_data.npz"
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
    def load_labeled_eeg_data(self, subjects, channels, 
                    labeled_eeg_data_dir=COLAB_LABELED_EEG_DIR,
                    epoch_start_ms=EPOCH_START_MS,
                    epoch_end_ms=EPOCH_END_MS,
                    pre_ms=PRE_WINDOW_MS,
                    post_ms=POST_WINDOW_MS,
                    fs=EEG_SAMPLE_FREQUENCY):
        """
        Load EEG data for specified subjects and channels.
        """
        print("\n" + "="*70)
        print("LOADING EEG DATA")
        print("="*70)
        
        all_X, all_y, all_subjects = [], [], []
        
        # Calculate epoch indices
        total_time_points = int((pre_ms + post_ms) / 1000 * fs)
        fixation_idx = int(pre_ms / 1000 * fs)
        epoch_start_idx = fixation_idx + int(epoch_start_ms / 1000 * fs)
        epoch_end_idx = fixation_idx + int(epoch_end_ms / 1000 * fs)
        
        print(f"\nEpoch settings:")
        print(f"  Window: [-{pre_ms}, {post_ms}] ms")
        print(f"  Epoch: [{epoch_start_ms}, {epoch_end_ms}] ms (indices {epoch_start_idx}:{epoch_end_idx})")
        print(f"  Sampling rate: {fs} Hz")
        
        # Load data from each subject
        print(f"\nLoading from {len(subjects)} subjects...")
        for subject in subjects:
            file_path = labeled_eeg_data_dir / subject / f"{pre_ms}_{post_ms}_training_data.npz"
            
            if not file_path.exists():
                print(f"  ⚠ {subject}: File not found")
                continue
            
            # Load data with memory mapping
            with np.load(file_path, mmap_mode='r') as data:
                ch_names = data['channel_names']
                
                # Find channel indices
                ch_indices = [list(ch_names).index(ch) for ch in channels if ch in ch_names]
                
                if len(ch_indices) == 0:
                    print(f"  ⚠ {subject}: No matching channels found")
                    continue
                
                # Load only needed data
                X_selected = data['X'][:, ch_indices, epoch_start_idx:epoch_end_idx].copy().astype(np.float32)
                y = data['y'][:]
            
            all_X.append(X_selected)
            all_y.append(y)
            all_subjects.extend([subject] * len(y))
            
            del X_selected
            gc.collect()
        
        # Combine all subjects
        X_combined = np.concatenate(all_X, axis=0).astype(np.float32)
        y_combined = np.concatenate(all_y, axis=0)
        del all_X, all_y
        gc.collect()
        
        # Print summary
        print(f"\n✓ Data loaded:")
        print(f"  Total trials: {X_combined.shape[0]}")
        print(f"  From {len(set(all_subjects))} subjects")
        print(f"  Channels: {len(ch_indices)}")
        print(f"  Time points: {X_combined.shape[2]}")
        print(f"  Duration: {(epoch_end_ms - epoch_start_ms)} ms")
        
        # Print label distribution
        label_counts = Counter(y_combined)
        total_samples = len(y_combined)
        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            proportion = count / total_samples
            print(f"  Label {label}: {count} samples ({proportion:.2%})")
        
        return X_combined, y_combined, np.array(all_subjects)

    @classmethod
    def load_stimulus_summary(self, subjects, pre_ms=PRE_WINDOW_MS, post_ms=POST_WINDOW_MS, labeled_eeg_data_dir=COLAB_LABELED_EEG_DIR):
        """
        Load stimulus summary for each trial.
        """
        print("\n" + "="*70)
        print("LOADING STIMULUS SUMMARY")
        print("="*70)
        
        all_stim = []
        
        print(f"\nLoading from {len(subjects)} subjects...")
        for subject in subjects:
            file_path = labeled_eeg_data_dir / subject / f"{pre_ms}_{post_ms}_training_data.npz"
            
            if not file_path.exists():
                print(f"  ⚠ {subject}: File not found")
                continue
            
            # Load stimulus data
            with np.load(file_path, mmap_mode='r') as data:
                if 'stimulus_summary' in data:
                    stim = data['stimulus_summary'][:]
                elif 'stimulus' in data:
                    stim = data['stimulus'][:]
                else:
                    print(f"  ⚠ {subject}: No stimulus data found")
                    continue
            
            all_stim.append(stim)
        
        # Combine all subjects
        stimulus_combined = np.concatenate(all_stim, axis=0)
        del all_stim
        gc.collect()
        
        print(f"\n✓ Stimulus summary loaded:")
        print(f"  Total samples: {stimulus_combined.shape[0]}")
        print(f"  Shape: {stimulus_combined.shape}")
        
        return stimulus_combined

    @classmethod
    def load_labeled_eeg_data_with_stimulus(self, subjects, channels,
                                epoch_start_ms=EPOCH_START_MS,
                                epoch_end_ms=EPOCH_END_MS,
                                pre_ms=PRE_WINDOW_MS,
                                post_ms=POST_WINDOW_MS,
                                fs=EEG_SAMPLE_FREQUENCY,
                                labeled_eeg_data_dir=COLAB_LABELED_EEG_DIR):
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
            file_path = labeled_eeg_data_dir / subject / f"{pre_ms}_{post_ms}_training_data.npz"
            
            if not file_path.exists():
                print(f"  ⚠ {subject}: File not found")
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


    # ============================================================================
    # USAGE EXAMPLES
    # ============================================================================

    """
    # Load only EEG data
    from config import SUBJECTS, DEFAULT_EEG_CHANNELS
    X, y, subjects = load_eeg_data(SUBJECTS, DEFAULT_EEG_CHANNELS)

    # Load EEG and stimulus together
    X, y, subjects, stimulus = load_eeg_data_with_stimulus(SUBJECTS, DEFAULT_EEG_CHANNELS)

    # Then use with split functions
    from data_split import split_normal
    X_train, X_val, X_test, y_train, y_val, y_test, stim_train, stim_val, stim_test = \
        split_normal(X, y, stimulus_summary=stimulus)
    """




