import os
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent / "data"

LABELED_ET_DATA_DIR = DATA_ROOT / "labeled_et_data"
LABELED_EEG_DATA_DIR = DATA_ROOT / "labeled_eeg_data"
RAW_DATA_ROOT = DATA_ROOT / "raw_data"
FEATURES_DIR = DATA_ROOT / "features"

ET_DIR = RAW_DATA_ROOT / "ET_clip"
IMAGE_DIR = RAW_DATA_ROOT / "Stimuli"
LABELS_DIR = RAW_DATA_ROOT / "label"
EEG_DIR = RAW_DATA_ROOT / "EEG_clean"
RAW_EEG_DIR = RAW_DATA_ROOT / "EEG"

TIME_SYNCH_PATH = RAW_DATA_ROOT / "Time Synchronization Data.xlsx"

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

STIM_WIDTH = 800
STIM_HEIGHT = 800

FIXATION_RADIUS = 50/2


EEG_SAMPLE_FREQUENCY = 500


COLUMN_NAMES = [
    "Time","Type","Trial",
    "L Raw X [px]","L Raw Y [px]","R Raw X [px]","R Raw Y [px]",
    "L Dia X [px]","L Dia Y [px]","L Pupil Diameter [mm]",
    "R Dia X [px]","R Dia Y [px]","R Pupil Diameter [mm]",
    "L CR1 X [px]","L CR1 Y [px]","L CR2 X [px]","L CR2 Y [px]",
    "R CR1 X [px]","R CR1 Y [px]","R CR2 X [px]","R CR2 Y [px]",
    "L POR X [px]","L POR Y [px]","R POR X [px]","R POR Y [px]",
    "Timing","Pupil Confidence","L Plane","R Plane",
    "L EPOS X","L EPOS Y","L EPOS Z","R EPOS X","R EPOS Y","R EPOS Z",
    "L GVEC X","L GVEC Y","L GVEC Z","R GVEC X","R GVEC Y","R GVEC Z",
    "Frame","Aux1","L Event Info","R Event Info","Stimulus"
]

ALL_SUBJECTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10",
    "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19", "P20",
    "P21", "P22", "P23", "P24", "P25", "P26", "P27", "P28", "P29", "P30",
    "P31", "P32", "P33", "P34", "P35", "P36", "P37", "P38"
]

# Subjects with good eyetracking data
# Subjects that were removed:
# More than 5% of trials didnt include fixations: 
# 'P03', 'P04', 'P08', 'P13', 'P18', 'P22'
ET_SUBJECTS = [
    "P01", "P02", "P04", "P05", "P06", "P07", "P09", "P10",
    "P11", "P12", "P14", "P15", "P16", "P17", "P19", "P20",
    "P21", "P23", "P24", "P25", "P26", "P27", "P28", "P29", "P30",
    "P31", "P32", "P33", "P34", "P35", "P36", "P37", "P38"
]

# Misaligned subjects: (P11, P15, P29) check again

# These subjects, have either misaligned data or very messy and flat EEG:
# P02, P04, P06, P11, P14, P15, P23, P24, P25, P26, P29, P32, P33, P35, P37


# P31 has to many invalid segments... missing data
# Too many segments removed: 2172 >= 50 for subject:P31

# P36 does not exist in the cleaned version

SUBJECTS = [
    "P01", "P05", "P07", "P09", "P10",
    "P12", "P16", "P17", "P19", "P20",
    "P21", "P27", "P28", "P30", 
    "P34", "P38"
]


# EEG Channel mapping (order matters - matches the EDF file channel order)
EEG_CHANNEL_NAMES = [
    'P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'T8', 'F8',
    'C4', 'F4', 'Fp2', 'Fz', 'C3', 'F3', 'Fp1', 'T7', 'F7', 'Oz', 
    'PO4', 'FC6', 'FC2', 'AF4', 'CP6', 'CP2', 'CP1', 'CP5', 'FC1', 'FC5', 
    'AF3', 'PO3'  # Note: 'Status' channel is excluded (not EEG data)
]

# Common channel selections for different analyses
FRONTAL_CHANNELS = ['F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2', 'AF3', 'AF4']
CENTRAL_CHANNELS = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6']
PARIETAL_CHANNELS = ['P3', 'P4', 'P7', 'P8', 'Pz', 'PO3', 'PO4']
OCCIPITAL_CHANNELS = ['O1', 'O2', 'Oz']
TEMPORAL_CHANNELS = ['T7', 'T8']

# Default channel selection for EEGNet (similar to original code)
DEFAULT_EEGNET_CHANNELS = ['F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6', 
                          'Fp1', 'Fp2', 'Pz', 'P3', 'P4', 'P7', 'P8']
