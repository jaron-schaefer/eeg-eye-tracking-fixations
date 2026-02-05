import numpy as np
import gc
from sklearn.model_selection import train_test_split

def split_normal(X, y, stimulus_summary=None, test_size=0.2, val_size=0.2, seed=42):
    """
    Random train/val/test split (works with any data).
    """
    has_stim = stimulus_summary is not None
    
    # First split: train+val vs test
    if has_stim:
        X_train, X_test, y_train, y_test, stim_train, stim_test = train_test_split(
            X, y, stimulus_summary, test_size=test_size, random_state=seed, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    
    del X  # Delete original
    gc.collect()
    
    # Second split: train vs val
    if has_stim:
        X_train, X_val, y_train, y_val, stim_train, stim_val = train_test_split(
            X_train, y_train, stim_train, test_size=val_size/(1-test_size), random_state=seed, stratify=y_train
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=seed, stratify=y_train
        )
    
    gc.collect()
    
    # Summary
    total_samples = len(y_train) + len(y_val) + len(y_test)
    print(f"Split data: {len(y_train)} train, {len(y_val)} val, {len(y_test)} test (total: {total_samples})")
    
    if has_stim:
        return X_train, X_val, X_test, y_train, y_val, y_test, stim_train, stim_val, stim_test
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test


def split_leave_n_out_generator(X, y, subjects, stimulus_summary=None, n_test_subjects=3, seed=42):
    """
    Leave-N-out cross-validation generator (yields one fold at a time for RAM efficiency).
    """
    has_stim = stimulus_summary is not None
    
    # Convert subjects to numpy array for boolean indexing
    subjects = np.array(subjects)
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    n_folds = n_subjects // n_test_subjects
    
    # Use fixed seed for reproducibility
    np.random.seed(seed)
    shuffled_subjects = np.random.permutation(unique_subjects)
    
    print(f"Leave-{n_test_subjects}-out CV: {n_folds} folds with {n_subjects} subjects")
    
    for fold_idx in range(n_folds):
        # Get test subjects for this fold
        test_subject_indices = np.arange(fold_idx * n_test_subjects, 
                                         (fold_idx + 1) * n_test_subjects)
        test_subjects = shuffled_subjects[test_subject_indices]
        train_val_subjects = np.setdiff1d(shuffled_subjects, test_subjects)
        
        # Create masks
        test_mask = np.isin(subjects, test_subjects)
        train_val_mask = np.isin(subjects, train_val_subjects)
        
        # Split by subject
        X_test = X[test_mask].copy()
        y_test = y[test_mask].copy()
        
        X_train_val = X[train_val_mask].copy()
        y_train_val = y[train_val_mask].copy()
        subj_train_val = subjects[train_val_mask]
        
        # Split stimulus summary if provided
        if has_stim:
            stim_test = stimulus_summary[test_mask].copy()
            stim_train_val = stimulus_summary[train_val_mask].copy()
        
        # Further split train_val (80/20)
        if has_stim:
            X_train, X_val, y_train, y_val, stim_train_val_split, stim_train, subj_train_split, subj_val = train_test_split(
                X_train_val, y_train_val, stim_train_val, subj_train_val, 
                test_size=0.2, random_state=seed, stratify=y_train_val
            )
            _, _, stim_val, _, _, _ = train_test_split(
                X_train_val, y_train_val, stim_train_val, 
                test_size=0.2, random_state=seed, stratify=y_train_val
            )
        else:
            X_train, X_val, y_train, y_val, _, _ = train_test_split(
                X_train_val, y_train_val, subj_train_val, 
                test_size=0.2, random_state=seed, stratify=y_train_val
            )
        
        del X_train_val, y_train_val, subj_train_val
        if has_stim:
            del stim_train_val
        gc.collect()
        
        # Yield fold as tuple to match split_normal format
        if has_stim:
            yield X_train, X_val, X_test, y_train, y_val, y_test, stim_train, stim_val, stim_test
        else:
            yield X_train, X_val, X_test, y_train, y_val, y_test, None, None, None

