import os
from typing import List
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from joblib import Parallel, delayed

from data.data_configs import *
from data.data_utils import transform_et_time_to_eeg_time


# --------------------------
# Utility functions
# --------------------------

def eye_valid(x, y, conf, w=SCREEN_WIDTH, h=SCREEN_HEIGHT):
    """Check if a gaze sample is valid."""
    if pd.isna(x) or pd.isna(y):
        return False
    if x == -1 or y == -1:
        return False
    if not (0 <= x <= w and 0 <= y <= h):
        return False
    if conf < 1:
        return False
    return True


def add_stimulus_coords(df):
    """Add coordinates relative to the stimulus center."""
    stim_x0 = (SCREEN_WIDTH - STIM_WIDTH) // 2
    stim_y0 = (SCREEN_HEIGHT - STIM_HEIGHT) // 2

    df["stim_L_x"] = df["L POR X [px]"] - stim_x0
    df["stim_L_y"] = df["L POR Y [px]"] - stim_y0
    df["stim_R_x"] = df["R POR X [px]"] - stim_x0
    df["stim_R_y"] = df["R POR Y [px]"] - stim_y0

    return df


def combine_gaze_with_event(row):
    """
    1. Calculates mean coordinates of Left/Right eyes (ignoring disparity/validity).
    2. Determines event label based on specific hierarchy:
       - Blink + (Any) -> Blink
       - Saccade + (Saccade or Fixation) -> Saccade
       - Fixation + Fixation -> Fixation
    """
    
    # --- 1. Coordinates: Always Mean ---
    # We take the average of stim_L and stim_R.
    # Note: If one eye is NaN, (NaN+Val)/2 is usually NaN. 
    # If you want to handle single-eye data, use np.nanmean, but here is strict mean:
    mean_x = (row['stim_L_x'] + row['stim_R_x']) / 2
    mean_y = (row['stim_L_y'] + row['stim_R_y']) / 2

    # --- 2. Event Logic Hierarchy ---
    evt_L = row['L Event Info']
    evt_R = row['R Event Info']
    
    event_label = 'Unknown' # Default

    # Priority 1: If either eye is blinking, it's a Blink
    if evt_L == 'Blink' or evt_R == 'Blink':
        event_label = 'Blink'

    # Priority 2: If either eye is Saccade (and not Blink), it's a Saccade
    # This covers: Saccade+Saccade AND Saccade+Fixation
    elif evt_L == 'Saccade' or evt_R == 'Saccade':
        event_label = 'Saccade'
        
    # Priority 3: Only if BOTH are Fixations, it's a Fixation
    elif evt_L == 'Fixation' and evt_R == 'Fixation':
        event_label = 'Fixation'
        
    return pd.Series([mean_x, mean_y, event_label], index=['mean_x', 'mean_y', 'event'])


def make_event_df(df):
    """Segment continuous events into start/end/duration and summary statistics."""
    df['event_change'] = df['event'] != df['event'].shift()
    df['event_id'] = df['event_change'].cumsum()
    events_list = []

    grouped = df.groupby('event_id')
    for _, g in grouped:
        event_type = g['event'].iloc[0]
        if pd.isna(event_type):
            continue  # skip undefined events

        event_dict = {
            'event_type': event_type,
            'start_time': g['Time'].iloc[0],
            'end_time': g['Time'].iloc[-1],
            'duration_ms': (g['Time'].iloc[-1] - g['Time'].iloc[0]) / 1000
        }

        if event_type == 'Fixation':
            event_dict['mean_x'] = g['mean_x'].mean()
            event_dict['mean_y'] = g['mean_y'].mean()
        elif event_type == 'Saccade':
            event_dict['start_x'] = g['mean_x'].iloc[0]
            event_dict['start_y'] = g['mean_y'].iloc[0]
            event_dict['end_x'] = g['mean_x'].iloc[-1]
            event_dict['end_y'] = g['mean_y'].iloc[-1]

        events_list.append(event_dict)

    return pd.DataFrame(events_list)


def get_plane_boxes(stimulus):
    """Parse object bounding boxes from XML labels."""
    label_path = os.path.join(LABELS_DIR, f"{stimulus}.xml")
    if not os.path.exists(label_path):
        print(f"Warning: label file for {stimulus} not found, skipping")
        return []

    tree = ET.parse(label_path)
    root = tree.getroot()

    plane_boxes = []
    for obj in root.findall('object'):
        if obj.find('name').text == 'airplane':
            bndbox = obj.find('bndbox')
            plane_boxes.append({
                'name': 'airplane',
                'bbox': (
                    int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text),
                    int(bndbox.find('ymax').text)
                )
            })
    return plane_boxes


def circle_overlaps_rectangle(cx, cy, r, rect):
    xmin, ymin, xmax, ymax = rect
    if xmin <= cx <= xmax and ymin <= cy <= ymax:
        return True
    closest_x = max(xmin, min(cx, xmax))
    closest_y = max(ymin, min(cy, ymax))
    return np.sqrt((cx - closest_x)**2 + (cy - closest_y)**2) <= r


def label_fixations(event_df, plane_boxes):
    """Label fixations that fall on airplane objects."""
    labels = []
    for _, row in event_df.iterrows():
        if row['event_type'] != 'Fixation' or pd.isna(row['mean_x']):
            labels.append(None)
            continue

        cx, cy = row['mean_x'], row['mean_y']
        overlap = any(circle_overlaps_rectangle(cx, cy, FIXATION_RADIUS, box['bbox']) for box in plane_boxes)
        labels.append(overlap)

    event_df['fixation_on_airplane'] = labels
    return event_df


def get_stimuli_for_subject(subject_dir, extension=".jpg.txt"):
    stimuli = [f.replace(extension, "") for f in os.listdir(subject_dir)
               if f.endswith(extension)]
    stimuli.sort()
    return stimuli


# --------------------------
# Process a single stimulus
# --------------------------
def process_stimulus_with_counts(subject, stimulus):
    # Skip gray stimuli
    if "gray" in stimulus.lower():
        return None

    et_data_path = os.path.join(ET_DIR, subject, f"{stimulus}.jpg.txt")
    df = pd.read_csv(et_data_path, comment='#', names=COLUMN_NAMES, low_memory=False)

    df = add_stimulus_coords(df)
    df[['mean_x', 'mean_y', 'event']] = df.apply(combine_gaze_with_event, axis=1, result_type='expand')
    event_df = make_event_df(df)

    plane_boxes = get_plane_boxes(stimulus)
    labeled_event_df = label_fixations(event_df, plane_boxes)
    labeled_event_df['stimulus'] = stimulus

    # Add stimulus onset column (first time in df)
    stim_onset_time = df['Time'].iloc[0]  # adjust 'time' column name if different
    labeled_event_df['stimulus_onset'] = stim_onset_time

    # Count airplanes and fixated airplanes
    n_airplanes = len(plane_boxes)
    n_fixated_airplanes = 0
    for box in plane_boxes:
        fixated = labeled_event_df.apply(
            lambda row: row['event_type'] == 'Fixation' and circle_overlaps_rectangle(
                row['mean_x'], row['mean_y'], FIXATION_RADIUS, box['bbox']
            ),
            axis=1
        ).any()
        if fixated:
            n_fixated_airplanes += 1

    summary = {
        'subject': subject,
        'stimulus': stimulus,
        'stimulus_onset': stim_onset_time,
        'n_airplanes': n_airplanes,
        'n_fixated_airplanes': n_fixated_airplanes,
        "n_fixations_on_airplane": labeled_event_df[
            (labeled_event_df['event_type'] == 'Fixation') &
            (labeled_event_df['fixation_on_airplane'] == True)
        ].shape[0],
        "n_fixations_not_on_airplane": labeled_event_df[
            (labeled_event_df['event_type'] == 'Fixation') &
            (labeled_event_df['fixation_on_airplane'] == False)
        ].shape[0]
    }

    return labeled_event_df, summary



# --------------------------
# Main function
# --------------------------
def label_et_data(subjects: List[str], n_jobs: int = 4):
    summary_list = []

    for subject in tqdm(subjects):
        subject_dir = os.path.join(ET_DIR, subject)
        stimuli = get_stimuli_for_subject(subject_dir)

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_stimulus_with_counts)(subject, stim) for stim in stimuli
        )

        # Filter out None (gray stimuli)
        results = [r for r in results if r is not None]

        if not results:
            continue

        all_subject_dfs = [r[0] for r in results]
        summary_list.extend([r[1] for r in results])

        subject_df = pd.concat(all_subject_dfs, ignore_index=True)

        # Vectorized EEG time transformation
        df_time_synch = pd.read_excel(TIME_SYNCH_PATH, engine="openpyxl")
        b_test_subj = df_time_synch.loc[df_time_synch['Participant No.'] == subject, 'b'].values[0]

        # Transform start/end times to EEG time
        subject_df["eeg_time_start"] = transform_et_time_to_eeg_time(subject_df["start_time"], b_test_subj)
        subject_df["eeg_end_start"] = transform_et_time_to_eeg_time(subject_df["end_time"], b_test_subj)

        # Transform stimulus onset to EEG time
        subject_df["eeg_stimulus_onset"] = transform_et_time_to_eeg_time(subject_df["stimulus_onset"], b_test_subj)

        # Also update the summary list with EEG times
        for summary in summary_list[-len(results):]:  # only the current subject/stimuli
            summary["eeg_stimulus_onset"] = transform_et_time_to_eeg_time(summary["stimulus_onset"], b_test_subj)

        save_path = os.path.join(LABELED_ET_ALL_DATA_DIR, f"{subject}_labeled_events.csv")
        subject_df.to_csv(save_path, index=False)

    # Save summary table for all subjects/stimuli
    summary_df = pd.DataFrame(summary_list)
    summary_path = os.path.join(LABELED_ET_ALL_DATA_DIR, "fixation_summary_all.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved fixation summary at {summary_path}")
