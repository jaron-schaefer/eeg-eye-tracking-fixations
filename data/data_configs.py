from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directories
DATA_ROOT = Path(__file__).resolve().parent / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw_data"

# Raw data directories
RAW_EEG_DIR = RAW_DATA_ROOT / "EEG"
EEG_DIR = RAW_DATA_ROOT / "EEG_clean data"
ET_DIR = RAW_DATA_ROOT / "ET_clip"
IMAGE_DIR = RAW_DATA_ROOT / "Stimuli"
LABELS_DIR = RAW_DATA_ROOT / "label"
TIME_SYNCH_PATH = RAW_DATA_ROOT / "Time Synchronization Data.xlsx"

# Labeled/processed data directories
LABELED_EEG_DATA_DIR = DATA_ROOT / "labeled_eeg_data"
LABELED_ET_DATA_DIR = DATA_ROOT / "labeled_et_data"
LABELED_ET_ALL_DATA_DIR = LABELED_ET_DATA_DIR / "all"
SELECTED_ET_DATA_DIR = LABELED_ET_DATA_DIR / "selected"

# Features and outputs
FEATURES_DIR = DATA_ROOT / "features"

# Create all directories
ALL_DIRS = [
    LABELED_EEG_DATA_DIR,
    LABELED_ET_DATA_DIR,
    LABELED_ET_ALL_DATA_DIR,
    SELECTED_ET_DATA_DIR,
    RAW_DATA_ROOT,
    FEATURES_DIR,
    ET_DIR,
    IMAGE_DIR,
    LABELS_DIR,
    EEG_DIR,
    RAW_EEG_DIR
]

for directory in ALL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# EYE TRACKING (ET) SETTINGS
# ============================================================================

# Screen and stimulus parameters
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
STIM_WIDTH = 800
STIM_HEIGHT = 800
FIXATION_RADIUS = 50

# ET data column names
ET_COLUMN_NAMES = [
    "Time", "Type", "Trial",
    "L Raw X [px]", "L Raw Y [px]", "R Raw X [px]", "R Raw Y [px]",
    "L Dia X [px]", "L Dia Y [px]", "L Pupil Diameter [mm]",
    "R Dia X [px]", "R Dia Y [px]", "R Pupil Diameter [mm]",
    "L CR1 X [px]", "L CR1 Y [px]", "L CR2 X [px]", "L CR2 Y [px]",
    "R CR1 X [px]", "R CR1 Y [px]", "R CR2 X [px]", "R CR2 Y [px]",
    "L POR X [px]", "L POR Y [px]", "R POR X [px]", "R POR Y [px]",
    "Timing", "Pupil Confidence", "L Plane", "R Plane",
    "L EPOS X", "L EPOS Y", "L EPOS Z", "R EPOS X", "R EPOS Y", "R EPOS Z",
    "L GVEC X", "L GVEC Y", "L GVEC Z", "R GVEC X", "R GVEC Y", "R GVEC Z",
    "Frame", "Aux1", "L Event Info", "R Event Info", "Stimulus"
]

# ============================================================================
# EEG SETTINGS
# ============================================================================

# Sampling frequency
EEG_SAMPLE_FREQUENCY = 500

# EEG channel names (order matters - matches EDF file channel order)
EEG_CHANNEL_NAMES = [
    'P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'T8', 'F8',
    'C4', 'F4', 'Fp2', 'Fz', 'C3', 'F3', 'Fp1', 'T7', 'F7', 'Oz',
    'PO4', 'FC6', 'FC2', 'AF4', 'CP6', 'CP2', 'CP1', 'CP5', 'FC1', 'FC5',
    'AF3', 'PO3'  # Note: 'Status' channel is excluded (not EEG data)
]

ARTIFACT_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F8',
    'AF3', 'AF4', 'T7', 'T8'
]

# Standard 10-20 system channel set
# ORIGINAL_CBRAMOD_CHANNELS = [
#     'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
#     'T3', 'C3', 'Cz', 'C4', 'T4',
#     'T5', 'P3', 'Pz', 'P4', 'T6',
#     'O1', 'O2'
# ]

CBRAMOD_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]


# Channel selections by region
CHANNEL_GROUPS = {
    'frontal': ['F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2', 'AF3', 'AF4'],
    'central': ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6'],
    'parietal': ['P3', 'P4', 'P7', 'P8', 'Pz', 'PO3', 'PO4'],
    'occipital': ['O1', 'O2', 'Oz'],
    'temporal': ['T7', 'T8'],
}

# Default channels for model training
DEFAULT_EEG_CHANNELS = ['F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6',
                        'Fp1', 'Fp2', 'Pz', 'P3', 'P4', 'P7', 'P8']

# Epoch extraction parameters
PRE_WINDOW_MS = 500      # ms before fixation
POST_WINDOW_MS = 1000    # ms after fixation
EPOCH_START_MS = -200    # epoch start relative to fixation
EPOCH_END_MS = 800       # epoch end relative to fixation

# ============================================================================
# SUBJECT INFORMATION
# ============================================================================

# All subjects in the dataset
ALL_SUBJECTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10",
    "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19", "P20",
    "P21", "P22", "P23", "P24", "P25", "P26", "P27", "P28", "P29", "P30",
    "P31", "P32", "P33", "P34", "P35", "P36", "P37", "P38"
]

# Subjects after quality control and cleaning
SUBJECTS = [
    "P01", "P07", "P09", "P10",
    "P16", "P17", "P20",
    "P21", "P27", "P28", "P30",
    "P34", "P38"
]

# ============================================================================
# BACKWARD COMPATIBILITY (for Colab)
# ============================================================================

# Override for Colab environment (adjust as needed)

COLAB_LABELED_EEG_DIR = Path('../labeled_eeg_data')



# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

TRAINING = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    'dropout': 0.1,
}

CBRAMOD = {
    'input_channels': 16,          # Number of EEG channels
    'time_segments': 10,            # Number of temporal segments
    'points_per_segment': 200,      # Time points per segment
}

# ============================================================================
# CROSS-VALIDATION SETTINGS
# ============================================================================

CV = {
    'method': 'leave_n_out',        # 'normal' or 'leave_n_out'
    'n_test_subjects': 3,           # For leave-n-out CV
    'test_size': 0.2,               # For normal CV
    'val_size': 0.2,                # For normal CV
    'random_seed': 42,
}

