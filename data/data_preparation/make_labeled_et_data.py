import os
from typing import List
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import math

from tqdm import tqdm
from data.data_configs import *
from data.data_utils import transform_et_time_to_eeg_time


def eye_valid(x, y, conf, w=SCREEN_WIDTH, h=SCREEN_HEIGHT):
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

    # top-left corner of stimulus in screen coordinates
    stim_x0 = (SCREEN_WIDTH - STIM_WIDTH) // 2
    stim_y0 = (SCREEN_HEIGHT - STIM_HEIGHT) // 2

    # Left eye
    df["stim_L_x"] = df["L POR X [px]"] - stim_x0
    df["stim_L_y"] = df["L POR Y [px]"] - stim_y0

    # Right eye
    df["stim_R_x"] = df["R POR X [px]"] - stim_x0
    df["stim_R_y"] = df["R POR Y [px]"] - stim_y0

    return df

def validity_reason(x, y, conf, w=SCREEN_WIDTH, h=SCREEN_HEIGHT):
    reasons = []
    if pd.isna(x) or pd.isna(y):
        reasons.append("NaN")
    if x == -1 or y == -1:
        reasons.append("Missing (-1)")
    if not (0 <= x <= w and 0 <= y <= h):
        reasons.append("Off-screen")
    if conf < 1:
        reasons.append("Low confidence")
    return ", ".join(reasons) if reasons else "Unknown"



def combine_gaze(row, disparity_thresh=80, debug=False):

    L_valid = eye_valid(row['stim_L_x'], row['stim_L_y'], row['Pupil Confidence'])
    R_valid = eye_valid(row['stim_R_x'], row['stim_R_y'], row['Pupil Confidence'])

    if L_valid and R_valid:
        dx = row['stim_L_x'] - row['stim_R_x']
        dy = row['stim_L_y'] - row['stim_R_y']

        if np.sqrt(dx*dx + dy*dy) > disparity_thresh:
            return pd.Series([np.nan, np.nan])

        return pd.Series([
            (row['stim_L_x'] + row['stim_R_x']) / 2,
            (row['stim_L_y'] + row['stim_R_y']) / 2
        ])

    elif L_valid:
        return pd.Series([row['stim_L_x'], row['stim_L_y']])

    elif R_valid:
        return pd.Series([row['stim_R_x'], row['stim_R_y']])

    else:
        if debug:
            raise ValueError(
                f"Invalid binocular sample at Time={row['Time']}. "
                f"Left eye: {validity_reason(row['stim_L_x'], row['stim_L_y'], row['Pupil Confidence'])}. "
                f"Right eye: {validity_reason(row['stim_R_x'], row['stim_R_y'], row['Pupil Confidence'])}."
            )
        return pd.Series([np.nan, np.nan])

def combine_events_both_eyes(df):

    def combined_event(row):
        if pd.isna(row['mean_x']) or pd.isna(row['mean_y']):
            return 'Blink'

        L = row['L Event Info']
        R = row['R Event Info']

        if L == 'Fixation' and R == 'Fixation':
            return 'Fixation'
        elif L == 'Saccade' and R == 'Saccade':
            return 'Saccade'
        else:
            return 'Blink'

    df[['mean_x', 'mean_y']] = df.apply(
        combine_gaze, axis=1, result_type='expand'
    )

    df['event'] = df.apply(combined_event, axis=1)

    return df


def make_event_df(df):
    # Add a dummy row at the end with a different event type
    dummy_row = df.iloc[-1].copy()
    dummy_row['event'] = 'DummyEvent'
    df = pd.concat([df, pd.DataFrame([dummy_row])], ignore_index=True)

    # Initialize current_type to the first row's event
    current_samples = [df.iloc[0]]
    current_type = df.iloc[0]['event']
    if current_type is None:
        raise ValueError("First row has undefined event (None).")

    events_list = []

    # Iterate starting from second row
    for idx, row in df.iloc[1:].iterrows():
        if row['event'] == current_type:
            current_samples.append(row)
        else:
            # Save previous event with type-specific logic
            temp_df = pd.DataFrame(current_samples)
            if current_type == 'Fixation':
                event_dict = {
                    'event_type': current_type,
                    'start_time': temp_df['Time'].iloc[0],
                    'end_time': temp_df['Time'].iloc[-1],
                    'duration_ms': (temp_df['Time'].iloc[-1] - temp_df['Time'].iloc[0]) / 1000,
                    'mean_x': temp_df['mean_x'].mean(),
                    'mean_y': temp_df['mean_y'].mean()
                }
            elif current_type == 'Saccade':
                event_dict = {
                    'event_type': current_type,
                    'start_time': temp_df['Time'].iloc[0],
                    'end_time': temp_df['Time'].iloc[-1],
                    'duration_ms': (temp_df['Time'].iloc[-1] - temp_df['Time'].iloc[0]) / 1000,
                    'start_x': temp_df['mean_x'].iloc[0],
                    'start_y': temp_df['mean_y'].iloc[0],
                    'end_x': temp_df['mean_x'].iloc[-1],
                    'end_y': temp_df['mean_y'].iloc[-1]
                }
            else:  # Blink or any other
                event_dict = {
                    'event_type': current_type,
                    'start_time': temp_df['Time'].iloc[0],
                    'end_time': temp_df['Time'].iloc[-1],
                    'duration_ms': (temp_df['Time'].iloc[-1] - temp_df['Time'].iloc[0]) / 1000
                }

            events_list.append(event_dict)

            # Start new event
            current_samples = [row]
            current_type = row['event']

    # Convert to DataFrame and remove dummy rows
    events_df = pd.DataFrame(events_list)

    return events_df


def get_plane_boxes(stimulus):

    label_path = label_path = os.path.join(LABELS_DIR, f"{stimulus}.xml")
    # Parse the XML file
    tree = ET.parse(label_path)  # replace with your file path
    root = tree.getroot()

    # Check filename
    filename = root.find('filename').text
    assert filename.split(".")[0] == stimulus
    # Check image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    # assert width == STIM_WIDTH, f"Wrong dimension for {stimulus}"
    # assert height == STIM_HEIGHT, f"Wrong dimension for {stimulus}"

    # Get all objects and their bounding boxes
    plane_boxes = []
    
    # has_plane = False
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        if obj_name == "airplane":
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            plane_boxes.append({
                'name': obj_name,
                'bbox': (xmin, ymin, xmax, ymax)
            })
            #has_plane = True

    # if not has_plane:
    #     print(stimulus)

    return plane_boxes

def circle_overlaps_rectangle(cx, cy, r, rect):
    xmin, ymin, xmax, ymax = rect
    
    # 1. Circle center inside rectangle
    if xmin <= cx <= xmax and ymin <= cy <= ymax:
        return True
    
    # 2. Closest point on rectangle to circle center
    closest_x = max(xmin, min(cx, xmax))
    closest_y = max(ymin, min(cy, ymax))
    
    # Distance from circle center to closest point
    distance = math.sqrt((cx - closest_x)**2 + (cy - closest_y)**2)
    return distance <= r

def label_fixations(event_df, plane_boxes):
    labels = []
    for _, row in event_df.iterrows():
        if row['event_type'] != 'Fixation' or pd.isna(row['mean_x']):
            labels.append(None)
            continue
        
        cx, cy = row['mean_x'], row['mean_y']
        overlapping_objects = []
        for box in plane_boxes:
            if circle_overlaps_rectangle(cx, cy, FIXATION_RADIUS, box['bbox']):
                overlapping_objects.append(True)
        labels.append(True if overlapping_objects else False)
    
    event_df['fixation_on_airplane'] = labels
    return event_df

def get_stimuli_for_subject(subject_dir, extension=".jpg.txt"):
    """
    Return a list of stimuli IDs (filenames without extension) for a subject directory.
    """
    all_files = os.listdir(subject_dir)
    stimuli = []
    for f in all_files:
        if f.endswith(extension):
            # Remove extension to get stimulus ID
            stim_id = f.replace(extension, "")
            if stim_id == "gray":
                continue
            stimuli.append(stim_id)
    stimuli.sort()  # optional, to keep consistent order
    return stimuli


def make_labeled_event_df(subjects: List[str]):
    for subject in tqdm(subjects):

        # get all stimuli the subject had
        subject_dir = os.path.join(ET_DIR, subject)
        stimuli = get_stimuli_for_subject(subject_dir)

        all_subject_dfs = []  # Collect labeled event DFs for all stimuli
        for stimulus in stimuli:
            # Build path to event data
            et_data_path = os.path.join(ET_DIR, subject, f"{stimulus}.jpg.txt")
            
            # Load event data
            df = pd.read_csv(et_data_path, comment='#', names=COLUMN_NAMES, low_memory=False)
            
            # Preprocess / combine events
            df = add_stimulus_coords(df.copy())
            df = combine_events_both_eyes(df.copy())
            event_df = make_event_df(df.copy())
            
            # Get object bounding boxes
            plane_boxes = get_plane_boxes(stimulus)
            
            # Label fixations
            labeled_event_df = label_fixations(event_df, plane_boxes)
            
            labeled_event_df['stimulus'] = stimulus
            
            # Append to subject list
            all_subject_dfs.append(labeled_event_df)
        
        # Combine all stimulus DFs into one long subject DF
        subject_df = pd.concat(all_subject_dfs, ignore_index=True)

        # get and apply time difference parameter b for the subject
        df_time_synch = pd.read_excel(TIME_SYNCH_PATH, engine="openpyxl")
        b_test_subj = df_time_synch.loc[df_time_synch['Participant No.'] == subject, 'b'].values[0]
        subject_df["eeg_time_start"] = subject_df["start_time"].apply(lambda x: transform_et_time_to_eeg_time(x, b_test_subj))
        subject_df["eeg_end_start"] = subject_df["end_time"].apply(lambda x: transform_et_time_to_eeg_time(x, b_test_subj))

        # Save subject DF
        save_path = os.path.join(LABELED_ET_DATA_DIR, f"{subject}_labeled_events.csv")
        subject_df.to_csv(save_path, index=False)
        print(f"Saved labeled event DF for subject {subject} at {save_path}")

if __name__ == "__main__":
    make_labeled_event_df(ALL_SUBJECTS[30:31])