from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyeeg
from pathlib import Path
from scipy import stats

from data.data_configs import EEG_CHANNEL_NAMES, FEATURES_DIR, SUBJECTS
from data.data_loader import DataLoader

FEATURE_NAMES = [
    'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
    'petrosian_fd', 'hjorth_mobility', 'hjorth_complexity', 
    'higuchi_fd', 'dfa', 'skewness', 'kurtosis',
    'minimum', 'maximum', 'std'
]

def extract_features_for_epoch(signal: np.ndarray, sampling_rate: float) -> List[float]:
    
    # Frequency band boundaries (delta..gamma): single bin_power call
    boundaries = [0.5, 4, 8, 13, 30, 45]
    features: List[float] = []
    
    # 1-5: Power spectral intensity for frequency bands
    band_power, _ = pyeeg.bin_power(signal, boundaries, sampling_rate)
    features.extend([float(p) for p in np.asarray(band_power).ravel()])
    
    # 6: Petrosian Fractal Dimension
    pfd = pyeeg.pfd(signal)
    features.append(float(pfd) if np.isscalar(pfd) else float(pfd[0]))
        
    # 7-8: Hjorth mobility and complexity
    hjorth = pyeeg.hjorth(signal)
    # Ensure we get exactly 2 scalar values
    if len(hjorth) >= 2:
        features.extend([float(hjorth[0]), float(hjorth[1])])
    else:
        features.extend([0.0, 0.0])
        
    # 9: Higuchi Fractal Dimension
    hfd = pyeeg.hfd(signal, 10)  # Kmax=10 as commonly used
    features.append(float(hfd) if np.isscalar(hfd) else float(hfd[0]))
        
    # 10: Detrended Fluctuation Analysis  
    dfa = pyeeg.dfa(signal)
    features.append(float(dfa) if np.isscalar(dfa) else float(dfa[0]))
        
    # 11: Skewness
    skew = stats.skew(signal)
    features.append(float(skew))
        
    # 12: Kurtosis
    kurt = stats.kurtosis(signal)
    features.append(float(kurt))
        
    # 13-15: Minimum, Maximum, Standard deviation
    minimum = float(np.min(signal))
    maximum = float(np.max(signal))
    std_dev = float(np.std(signal))
    features.extend([minimum, maximum, std_dev])
    
    # Validate we have exactly 15 features
    if len(features) != 15:
        raise ValueError(f"Expected 15 features, got {len(features)}: {features}")
    
    return features


def extract_features_for_channel(channel_data: np.ndarray, sampling_rate: float, ch_name: str) -> Tuple[np.ndarray, List[str]]:

    n_epochs = channel_data.shape[0] # n_epochs are how many fixations a subject made
    channel_features = []
    
    for epoch_idx in range(n_epochs):
        signal = channel_data[epoch_idx]
        epoch_features = extract_features_for_epoch(signal, sampling_rate)
        channel_features.append(epoch_features)

    feature_names = [f'{ch_name}_{name}' for name in FEATURE_NAMES]
    
    return np.array(channel_features), feature_names



def extract_features_for_subject(data: np.ndarray, sampling_rate: float, ch_names: List[str]) -> Tuple[np.ndarray, List[str]]:

    n_epochs, n_channels, n_times = data.shape
    assert len(ch_names) == n_channels
    
    all_features = []
    all_feature_names = []
    
    for ch_idx in tqdm(range(n_channels)):

        ch_name = ch_names[ch_idx]
        channel_data = data[:, ch_idx, :] # All epochs for this channel
        
        ch_features, feature_names = extract_features_for_channel(
            channel_data, sampling_rate, ch_name
        )
        
        # For first channel, initialize the feature matrix
        if ch_idx == 0:
            all_features = ch_features
        else:
            # Concatenate features horizontally (add more columns)
            all_features = np.concatenate([all_features, ch_features], axis=1)
        
        all_feature_names.extend(feature_names)
    
    return all_features, all_feature_names



def extract_and_save_subject_features(subject: str, channels: List[str]) -> Tuple[pd.DataFrame, np.ndarray, Path]:
    
    X, y, channel_names, metadata = DataLoader.load_subject_labeled_eeg(subject, channels)
    sfreq = float(metadata['sampling_rate'])

    all_features, all_names = extract_features_for_subject(X, sfreq, channel_names)

    output_dir = FEATURES_DIR / subject
    output_dir.mkdir(parents = True, exist_ok=True)

    df = pd.DataFrame(all_features, columns=all_names)
    df['label'] = np.squeeze(y)
    df.to_csv(output_dir / "features.csv", index=False)

    return df, y, output_dir


def extract_and_save_features_for_subjects(subjects: List[str] = SUBJECTS, channels: List[str] = EEG_CHANNEL_NAMES) -> None:
    
    for subject in sorted(subjects):
        try:
            df, labels, output_path = extract_and_save_subject_features(subject, channels)
        except Exception as e:
            print(f"  Error processing {subject}: {e}")


if __name__ == "__main__":
    extract_and_save_features_for_subjects(SUBJECTS[:1])
