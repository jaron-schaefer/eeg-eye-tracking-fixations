from data.download_dataset import download_dataset
from data.data_preparation.extract_eeg_features import extract_and_save_features_for_subjects
from data.data_preparation.extract_et_features import extract_et_features
from data.data_preparation.label_eeg_data import label_eeg_data
from data.data_preparation.label_et_data import label_et_data
from data.data_preparation.select_et_trials import select_et_trials
from data.data_configs import ALL_SUBJECTS, SUBJECTS


def main():
    """Run the complete data preparation pipeline"""
    print("=== DATA PREPARATION PIPELINE ===")

    subjects = SUBJECTS
    
    # Step 1: Download dataset (if needed)
    # download_dataset()
    
    # Step 2: Label eye tracking data
    print("\\n1. Labeling eye tracking data...")
    # label_et_data(subjects)
    
    # Step 3: Select eye tracking trials
    print("\\n2. Selecting eye tracking trials...")
    # select_et_trials(subjects)

    # Step 4: Label EEG data (uncomment to run)
    print("\\n3. Labeling EEG data...")
    # label_eeg_data(subjects)
    
    # Step 5: Extract EEG features (uncomment to run)
    print("\\n4. Extracting EEG features...")
    extract_and_save_features_for_subjects(subjects[8:])
    
    # Step 6: Process eye tracking features (uncomment to run)  
    print("\\n5. Processing eye tracking features...")
    extract_et_features(subjects)
    
    print("\\n=== DATA PREPARATION COMPLETE ===")


if __name__ == "__main__":
    main()