import os
import random
from typing import List, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Make sure package imports resolve the same way as in the notebook
from data.data_configs import (
    IMAGE_DIR,
    ET_DIR,
    LABELED_ET_ALL_DATA_DIR,
    ET_COLUMN_NAMES,
    FIXATION_RADIUS,
    SELECTED_ET_DATA_DIR,
    SELECTED_ET_DATA_DIR,
)
from data.data_preparation.label_et_data import (
    combine_gaze_with_event,
    add_stimulus_coords,
    get_plane_boxes,
)


def load_events(subject: str, selected: bool = False) -> pd.DataFrame:
    """Load labeled events CSV for a given subject."""
    dir_path = LABELED_ET_ALL_DATA_DIR if not selected else SELECTED_ET_DATA_DIR
    path = os.path.join(dir_path, f"{subject}_labeled_events.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labeled events file not found: {path}")
    return pd.read_csv(path, dtype={"stimulus": str})


def list_available_stimuli(subject: str) -> List[str]:
    """List stimuli IDs available for a given subject in ET clips."""
    subj_dir = os.path.join(ET_DIR, subject)
    if not os.path.isdir(subj_dir):
        return []
    stimuli: List[str] = []
    for name in os.listdir(subj_dir):
        # Expect files like <stimulus>.jpg.txt
        if name.endswith('.jpg.txt') and len(name) > len('.jpg.txt'):
            stimuli.append(name[:-len('.jpg.txt')])
    stimuli.sort()
    return stimuli


def pick_random_stimulus(events_df: pd.DataFrame, subject: str, seed: Optional[int] = None) -> Optional[str]:
    """Pick a random stimulus ID that exists for the subject in ET_clip."""
    if seed is not None:
        random.seed(seed)
    # Stimuli present in events
    event_stimuli = set(events_df["stimulus"].dropna().unique().tolist())
    # Stimuli actually recorded for the subject
    available_stimuli = set(list_available_stimuli(subject))
    # Intersection
    candidate_stimuli = list(event_stimuli & available_stimuli)
    if len(candidate_stimuli) == 0:
        return None
    return random.choice(candidate_stimuli)


def get_image_path(stimulus: str) -> str:
    """Return path to the stimulus image (.png)."""
    return os.path.join(IMAGE_DIR, f"{stimulus}.png")


def get_et_data_path(subject: str, stimulus: str) -> str:
    """Return path to the raw ET text file (.jpg.txt)."""
    return os.path.join(ET_DIR, subject, f"{stimulus}.jpg.txt")


def load_raw_et_df(subject: str, stimulus: str) -> pd.DataFrame:
    """
    Load raw ET data for a given subject and stimulus,
    add stimulus coordinates, and compute combined gaze and event labels.
    """
    et_path = get_et_data_path(subject, stimulus)
    if not os.path.exists(et_path):
        raise FileNotFoundError(f"ET data not found: {et_path}")

    # Load raw ET data
    raw_df = pd.read_csv(et_path, comment='#', names=ET_COLUMN_NAMES, low_memory=False)

    # Add coordinates relative to stimulus center
    raw_df = add_stimulus_coords(raw_df.copy())

    # Apply combined gaze and event labeling row-wise
    raw_df[['mean_x', 'mean_y', 'event']] = raw_df.apply(combine_gaze_with_event, axis=1, result_type='expand')

    raw_df.to_csv("debug_raw_et.csv", index=False)  # Debugging line

    return raw_df



def plot_gaze(df: pd.DataFrame, image_path: str, size=(8, 6), alpha=0.6, point_size=10):
    """Plot gaze points over the stimulus image using mean coordinates."""
    img = Image.open(image_path)
    img_width, img_height = img.size

    plt.figure(figsize=size)
    # the image needs to be turned around so it mateches the usualy top left origin of the eye tracking and boxes
    plt.imshow(img, extent=[0, img_width, img_height, 0])
    plt.scatter(df['mean_x'], df['mean_y'], alpha=alpha, s=point_size, c='red')

    plt.xlim(-100, 900)
    plt.ylim(-100, 900)
    plt.xlabel("X [px]")
    plt.ylabel("Y [px]")
    plt.title("Gaze on Stimulus")
    plt.tight_layout()
    plt.show()


def plot_events_on_image(events_df: pd.DataFrame, image_path: str, objects: List[Dict],
                         size=(8, 6), fixation_radius: int = FIXATION_RADIUS):
    """Plot fixation circles and object rectangles on the stimulus image."""
    img = Image.open(image_path)
    img_width, img_height = img.size

    plt.figure(figsize=size)
    plt.imshow(img, extent=[0, img_width, img_height, 0])

    from matplotlib.patches import Rectangle

    for obj in objects:
        xmin, ymin, xmax, ymax = obj['bbox']
        width = xmax - xmin
        height = ymax - ymin

        rectangle = Rectangle((xmin, ymin), width, height,
                              linewidth=2, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rectangle)

    fix = events_df[events_df['event_type'] == 'Fixation']
    for _, row in fix.iterrows():
        color = "green" if bool(row.get("fixation_on_airplane", False)) else "red"
        circle = plt.Circle(
            (row['mean_x'], row['mean_y']),
            fixation_radius,
            color=color,
            alpha=0.35,
            linewidth=1
        )
        plt.gca().add_patch(circle)

    # Compute fixation counts (total, on-target, off-target)
    total_fix = int(len(fix))
    # Robustly handle missing column by defaulting to False
    fixation_on_col = (
        fix['fixation_on_airplane']
        if 'fixation_on_airplane' in fix.columns
        else pd.Series(False, index=fix.index)
    )
    on_target = int(fixation_on_col.fillna(False).astype(bool).sum())
    off_target = int(total_fix - on_target)

    # Print counts to console for traceability
    print(f"Fixations: total={total_fix}, on-target={on_target}, off-target={off_target}")

    # Annotate counts on the plot for quick visual reference
    annotation = (
        f"Fixations\nTotal: {total_fix}\nOn-target: {on_target}\nOff-target: {off_target}"
    )
    plt.text(
        10,
        20,
        annotation,
        fontsize=10,
        color='white',
        bbox=dict(facecolor='black', alpha=0.5, pad=4)
    )

    plt.ylim(-100, 900)
    plt.xlim(-100, 900)
    plt.xlabel("Stimulus X [px]")
    plt.ylabel("Stimulus Y [px]")
    plt.title("Fixations and Saccades on Stimulus")
    plt.tight_layout()
    plt.show()


def plot_stimulus_for_subject(
    subject: str,
    also_gaze: bool = False,
    stimulus: Optional[str] = None,
    seed: Optional[int] = None,
    selected = False
):
    """Plot labeled events (and optionally gaze) for a subject.

    If `stimulus` is provided, use it (after validating presence in both the
    labeled events and the subject's ET clips). Otherwise, pick a random
    available stimulus for the subject and print which one was chosen.
    """
    events = load_events(subject, selected=selected)

    if stimulus is None:
        stim = pick_random_stimulus(events, subject, seed=seed)
        if stim is None:
            print(f"No available stimuli for subject {subject} in ET_clip.")
            return
        print(f"Selected random stimulus for {subject}: {stim}")
    else:
        stim = str(stimulus)
        et_path = get_et_data_path(subject, stim)
        in_events = (events["stimulus"] == stim).any()
        in_et = os.path.exists(et_path)
        if not (in_events and in_et):
            missing_parts = []
            if not in_events:
                missing_parts.append("labeled events")
            if not in_et:
                missing_parts.append("ET_clip file")
            print(
                f"Stimulus '{stim}' unavailable for subject {subject} (missing: {', '.join(missing_parts)})."
            )
            return

    image_path = get_image_path(stim)
    plane_boxes = get_plane_boxes(stim)
    events_for_stim = events[events["stimulus"] == stim]

    events_for_stim.to_csv("debug_raw.csv", index=False) 

    plot_events_on_image(events_for_stim, image_path, plane_boxes)

    if also_gaze:
        try:
            raw_df = load_raw_et_df(subject, stim)
            plot_gaze(raw_df, image_path)
        except FileNotFoundError as e:
            print(str(e))