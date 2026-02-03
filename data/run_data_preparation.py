from data.data.download_dataset import download_dataset
from data.data_preparation.extract_eeg_features import extract_and_save_features_for_subjects
from data.data_preparation.extract_et_features import process_eye_tracking_data
from data.data_preparation.label_eeg_data import label_eeg_data
from data.data_preparation.label_et_data import label_et_data
from data.data_preparation.select_et_trials import select_et_trials
from data.data_configs import ALL_SUBJECTS, DEFAULT_EEG_CHANNELS, SUBJECTS


def main():
    # download_dataset()
    label_et_data(SUBJECTS)
    select_et_trials(SUBJECTS)

    # Do only for selected subjects

    #label_eeg_data(subjects=SUBJECTS)
    #extract_and_save_features_for_subjects(SUBJECTS[:1], DEFAULT_EEG_CHANNELS, mode="simple")
    #extract_and_save_features_for_subjects(SUBJECTS[:1], DEFAULT_EEG_CHANNELS, mode="advanced")
    #process_eye_tracking_data(subjects=SUBJECTS[:1])

if __name__ == "__main__":
    main()