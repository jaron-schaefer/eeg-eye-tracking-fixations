import numpy as np
import gc
from sklearn.model_selection import train_test_split

def split_normal(X, y, stimulus_summary=None, test_size=0.2, val_size=0.2, seed=42):
    """
    Random train/val/test split (works with any data).
    """
    print("\n" + "="*70)
    print("RANDOM TRAIN/VAL/TEST SPLIT")
    print("="*70)
    
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
    
    # Print stats
    print(f"\nTrain: {X_train.shape[0]} samples ({X_train.shape[0]/len(y)*100:.1f}%)")
    print(f"  {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    print(f"\nVal:   {X_val.shape[0]} samples ({X_val.shape[0]/len(y)*100:.1f}%)")
    print(f"  {dict(zip(*np.unique(y_val, return_counts=True)))}")
    
    print(f"\nTest:  {X_test.shape[0]} samples ({X_test.shape[0]/len(y)*100:.1f}%)")
    print(f"  {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    if has_stim:
        return X_train, X_val, X_test, y_train, y_val, y_test, stim_train, stim_val, stim_test
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test


def split_leave_n_out_generator(X, y, subjects, stimulus_summary=None, n_test_subjects=3, seed=42):
    """
    Leave-N-out cross-validation generator (yields one fold at a time for RAM efficiency).
    """
    print("\n" + "="*70)
    print(f"LEAVE-{n_test_subjects}-OUT CROSS-VALIDATION")
    print("="*70)
    
    has_stim = stimulus_summary is not None
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    n_folds = n_subjects // n_test_subjects
    
    print(f"Total subjects: {n_subjects}")
    print(f"Folds: {n_folds} (each with {n_test_subjects} test subjects)")
    
    # Use fixed seed for reproducibility
    np.random.seed(seed)
    shuffled_subjects = np.random.permutation(unique_subjects)
    
    for fold_idx in range(n_folds):
        print(f"\n{'─'*70}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'─'*70}")
        
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
        
        # Print fold stats
        print(f"Test subjects: {list(test_subjects)}")
        print(f"\nTrain: {X_train.shape[0]} samples")
        print(f"  {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"\nVal:   {X_val.shape[0]} samples")
        print(f"  {dict(zip(*np.unique(y_val, return_counts=True)))}")
        print(f"\nTest:  {X_test.shape[0]} samples")
        print(f"  {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        # Build fold dictionary
        fold_dict = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'fold_idx': fold_idx,
            'test_subjects': test_subjects
        }
        
        # Add stimulus summary if provided
        if has_stim:
            fold_dict['stim_train'] = stim_train
            fold_dict['stim_val'] = stim_val
            fold_dict['stim_test'] = stim_test
        
        # Yield fold
        yield fold_dict


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Example 1: NORMAL SPLIT (without stimulus_summary)
# X_train, X_val, X_test, y_train, y_val, y_test = split_normal(X_shaped, y_all)
# results = train_model(X_train, y_train, X_val, y_val, X_test, y_test)

# Example 2: NORMAL SPLIT (with stimulus_summary)
# X_train, X_val, X_test, y_train, y_val, y_test, stim_train, stim_val, stim_test = \
#     split_normal(X_shaped, y_all, stimulus_summary=stimulus_summary_data)
# results = train_model(X_train, y_train, X_val, y_val, X_test, y_test, stim_train, stim_val, stim_test)

# Example 3: LEAVE-N-OUT (without stimulus_summary)
# fold_generator = split_leave_n_out_generator(X_shaped, y_all, subjects_all, n_test_subjects=3)
# fold_results = []
# for fold in fold_generator:
#     results = train_model(fold['X_train'], fold['y_train'], 
#                          fold['X_val'], fold['y_val'],
#                          fold['X_test'], fold['y_test'])
#     fold_results.append(results)
#     del fold
#     gc.collect()

# Example 4: LEAVE-N-OUT (with stimulus_summary)
# fold_generator = split_leave_n_out_generator(X_shaped, y_all, subjects_all, 
#                                             stimulus_summary=stimulus_summary_data, 
#                                             n_test_subjects=3)
# fold_results = []
# for fold in fold_generator:
#     results = train_model(fold['X_train'], fold['y_train'], 
#                          fold['X_val'], fold['y_val'],
#                          fold['X_test'], fold['y_test'],
#                          fold['stim_train'], fold['stim_val'], fold['stim_test'])
#     fold_results.append(results)
#     del fold
#     gc.collect()