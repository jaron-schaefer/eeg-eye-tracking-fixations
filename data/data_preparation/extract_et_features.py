import os
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict

from data.data_configs import *
from data.data_utils import transform_eeg_time_to_ms


def load_fixations(subject: str) -> pd.DataFrame:
    """Load labeled fixation events for a subject from CSV file."""
    events_path = os.path.join(LABELED_ET_ALL_DATA_DIR, f"{subject}_labeled_events.csv")
    df = pd.read_csv(events_path, dtype={"stimulus": str})
    # Load ALL events (both Fixations and Saccades) for feature extraction
    all_events = df.sort_values(by="eeg_time_start")
    
    # Check for NaN in fixation labels
    fixations = all_events[all_events["event_type"] == "Fixation"]
    if fixations["fixation_on_airplane"].isna().any():
        raise ValueError(f"NaN values in 'fixation_on_airplane' for {subject}")
    
    return all_events  # Return all events, not just fixations


def extract_pupil_features(event_data: pd.DataFrame, prefix: str = '') -> Dict:
    """Extract pupil-related features."""
    features = {}
    features[f'{prefix}mean_pupil_left'] = event_data["L Pupil Diameter [mm]"].mean()
    features[f'{prefix}mean_pupil_right'] = event_data["R Pupil Diameter [mm]"].mean()
    features[f'{prefix}mean_pupil_avg'] = (features[f'{prefix}mean_pupil_left'] + features[f'{prefix}mean_pupil_right']) / 2
    features[f'{prefix}std_pupil_left'] = event_data["L Pupil Diameter [mm]"].std()
    features[f'{prefix}std_pupil_right'] = event_data["R Pupil Diameter [mm]"].std()
    
    pupil_avg = (event_data["L Pupil Diameter [mm]"] + event_data["R Pupil Diameter [mm]"]) / 2
    features[f'{prefix}min_pupil_avg'] = pupil_avg.min()
    features[f'{prefix}max_pupil_avg'] = pupil_avg.max()
    features[f'{prefix}pupil_range'] = features[f'{prefix}max_pupil_avg'] - features[f'{prefix}min_pupil_avg']
    
    return features

def extract_gaze_stability_features(event_data: pd.DataFrame, prefix: str = '') -> Dict:
    """Extract gaze stability features (without position)."""
    features = {}
    # Dispersion/stability measures only (not mean position)
    features[f'{prefix}std_gaze_x'] = event_data["L POR X [px]"].std()
    features[f'{prefix}std_gaze_y'] = event_data["L POR Y [px]"].std()
    features[f'{prefix}gaze_dispersion'] = np.sqrt(features[f'{prefix}std_gaze_x']**2 + features[f'{prefix}std_gaze_y']**2)
    
    # Range of gaze movement during fixation
    features[f'{prefix}gaze_range_x'] = event_data["L POR X [px]"].max() - event_data["L POR X [px]"].min()
    features[f'{prefix}gaze_range_y'] = event_data["L POR Y [px]"].max() - event_data["L POR Y [px]"].min()
    
    return features

def extract_eye_movement_features(event_data: pd.DataFrame, prefix: str = '') -> Dict:
    """Extract eye movement dynamics features."""
    features = {}
    # Gaze vector features (eye movement direction, not position)
    features[f'{prefix}mean_gvec_x'] = event_data["L GVEC X"].mean()
    features[f'{prefix}mean_gvec_y'] = event_data["L GVEC Y"].mean()
    features[f'{prefix}mean_gvec_z'] = event_data["L GVEC Z"].mean()
    features[f'{prefix}std_gvec_x'] = event_data["L GVEC X"].std()
    features[f'{prefix}std_gvec_y'] = event_data["L GVEC Y"].std()
    features[f'{prefix}std_gvec_z'] = event_data["L GVEC Z"].std()
    
    # Eye position variability (depth/distance changes)
    features[f'{prefix}std_epos_z'] = event_data["L EPOS Z"].std()
    
    return features

def extract_saccade_features(saccade_data: pd.DataFrame, saccade_start: float, 
                            saccade_end: float, fixation_time: float, 
                            prefix: str) -> Dict:
    """Extract features from a saccade event."""
    features = {}
    features[f'{prefix}_duration'] = saccade_end - saccade_start
    features[f'{prefix}_iei'] = abs(fixation_time - saccade_end)
    
    if len(saccade_data) > 0:
        start_x = saccade_data.iloc[0]["L POR X [px]"]
        start_y = saccade_data.iloc[0]["L POR Y [px]"]
        end_x = saccade_data.iloc[-1]["L POR X [px]"]
        end_y = saccade_data.iloc[-1]["L POR Y [px]"]
        
        # Amplitude and velocity (not direction, as that would reveal position)
        features[f'{prefix}_amplitude'] = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        features[f'{prefix}_velocity'] = features[f'{prefix}_amplitude'] / features[f'{prefix}_duration'] if features[f'{prefix}_duration'] > 0 else 0
        
        # Peak velocity during saccade
        if len(saccade_data) > 1:
            velocities = []
            for i in range(1, len(saccade_data)):
                dx = saccade_data.iloc[i]["L POR X [px]"] - saccade_data.iloc[i-1]["L POR X [px]"]
                dy = saccade_data.iloc[i]["L POR Y [px]"] - saccade_data.iloc[i-1]["L POR Y [px]"]
                dt = saccade_data.iloc[i]["timestamp"] - saccade_data.iloc[i-1]["timestamp"]
                if dt > 0:
                    velocities.append(np.sqrt(dx**2 + dy**2) / dt)
            features[f'{prefix}_peak_velocity'] = max(velocities) if velocities else np.nan
        else:
            features[f'{prefix}_peak_velocity'] = np.nan
        
        # Pupil size during saccade
        pupil_avg = (saccade_data["L Pupil Diameter [mm]"] + saccade_data["R Pupil Diameter [mm]"]) / 2
        features[f'{prefix}_mean_pupil'] = pupil_avg.mean()
    else:
        for key in ['amplitude', 'velocity', 'peak_velocity', 'mean_pupil']:
            features[f'{prefix}_{key}'] = np.nan
    
    return features

def get_null_saccade_features(prefix: str) -> Dict:
    """Return NaN features when no saccade exists."""
    return {
        f'{prefix}_duration': np.nan,
        f'{prefix}_iei': np.nan,
        f'{prefix}_amplitude': np.nan,
        f'{prefix}_velocity': np.nan,
        f'{prefix}_peak_velocity': np.nan,
        f'{prefix}_mean_pupil': np.nan
    }

def extract_fixation_features(fix_df: pd.DataFrame, raw_df: pd.DataFrame, 
                              subject: str, stimulus: str) -> pd.DataFrame:
    """Extract comprehensive features for each fixation (position-independent, context-free)."""
    features_list = []
    fix_df_sorted = fix_df.sort_values('start_time').reset_index(drop=True)
    
    for idx, row in fix_df_sorted.iterrows():
        # Only extract features FOR fixations, but we need saccades in the df
        if row["event_type"] != "Fixation":
            continue
        
        event_start = row["start_time"]
        event_end = row["end_time"]
        event_data = raw_df[(raw_df["timestamp"] >= event_start) & (raw_df["timestamp"] <= event_end)].copy()
        
        if len(event_data) == 0:
            continue
        
        # Initialize features dict - ONLY metadata for identification, not for training
        features = {
            'subject': subject,
            'stimulus': stimulus,
            'fixation_duration': event_end - event_start,
        }
        
        # Extract fixation features (no position information)
        features.update(extract_pupil_features(event_data))
        features.update(extract_gaze_stability_features(event_data))
        features.update(extract_eye_movement_features(event_data))
        
        # Extract saccade before features
        if idx > 0 and fix_df_sorted.iloc[idx - 1]["event_type"] == "Saccade":
            prev_event = fix_df_sorted.iloc[idx - 1]
            saccade_data = raw_df[(raw_df["timestamp"] >= prev_event["start_time"]) & 
                                 (raw_df["timestamp"] <= prev_event["end_time"])]
            features.update(extract_saccade_features(
                saccade_data, prev_event["start_time"], prev_event["end_time"], 
                event_start, 'saccade_before'
            ))
        else:
            features.update(get_null_saccade_features('saccade_before'))
        
        # Extract saccade after features
        if idx < len(fix_df_sorted) - 1 and fix_df_sorted.iloc[idx + 1]["event_type"] == "Saccade":
            next_event = fix_df_sorted.iloc[idx + 1]
            saccade_data = raw_df[(raw_df["timestamp"] >= next_event["start_time"]) & 
                                 (raw_df["timestamp"] <= next_event["end_time"])]
            features.update(extract_saccade_features(
                saccade_data, next_event["start_time"], next_event["end_time"], 
                event_end, 'saccade_after'
            ))
        else:
            features.update(get_null_saccade_features('saccade_after'))
        
        features['fixation_on_airplane'] = row['fixation_on_airplane']
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def process_eye_tracking_data(subjects: List[str], et_dir: str = ET_DIR, column_names: List[str] = COLUMN_NAMES,
                              output_dir: str = 'data/data/features/et_features') -> Dict[str, pd.DataFrame]:

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    subject_features = {}
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_feature_list = []
        all_events_df = load_fixations(subject)  # Now contains both fixations and saccades
        
        for stimulus in all_events_df["stimulus"].unique():
            trial_events = all_events_df[all_events_df["stimulus"] == stimulus]
            
            et_data_path = os.path.join(et_dir, subject, f"{stimulus}.jpg.txt")
            
            try:
                raw_df = pd.read_csv(et_data_path, comment='#', names=column_names, low_memory=False)
                
                # Rename timestamp column if needed
                if 'Time' in raw_df.columns:
                    raw_df = raw_df.rename(columns={'Time': 'timestamp'})
                
                # Extract features for this trial
                trial_features = extract_fixation_features(trial_events, raw_df, subject, stimulus)
                subject_feature_list.append(trial_features)
                
            except Exception as e:
                print(f"    Error processing {stimulus}: {e}")
                continue
        
        # Combine features for this subject
        if subject_feature_list:
            subject_df = pd.concat(subject_feature_list, ignore_index=True)
            subject_features[subject] = subject_df
            
            # Save to individual file
            output_path = os.path.join(output_dir, f'{subject}_features.csv')
            subject_df.to_csv(output_path, index=False)
            print(f"  Saved {len(subject_df)} fixations for {subject} with {len(subject_df.columns)} features")
        else:
            print(f"  No features extracted for {subject}\n")
    
    print(f"All features saved to {output_dir}/")
    return subject_features