"""
Model Optimization Runner (Simplified)

Optimized for speed and minimal CPU RAM usage:
- Always uses balanced class weights (no optimization needed)
- Aggressive CPU memory cleanup after each model
- Simple immediate saving after each model (no crash recovery complexity)
"""

import numpy as np
import pandas as pd
import json
import gc
import torch
import psutil
from pathlib import Path
from sklearn.model_selection import ParameterGrid, train_test_split
from datetime import datetime

# Import models
from classification.models.eeg_net.eegnet_classifier import EEGNetClassifier
from classification.models.cbramod.cbramod_classifier import CBraModClassifier
from classification.models.csp.csp_classifier import CSPClassifier
from classification.models.features.feature_classifier import FeatureClassifier
from classification.models.features.et_feature_classifier import ETFeatureClassifier

# Import data utilities
from data.data_loader import DataLoader
from data.data_configs import CBRAMOD_CHANNELS, SUBJECTS, DEFAULT_EEG_CHANNELS

def get_data_path():
    """Get data path - for Colab, data is next to repository folder."""
    current_dir = Path(__file__).parent.absolute()
    if '/content/' in str(current_dir):
        return Path('/content/drive/MyDrive/eye-eeg')
    else:
        return current_dir.parent / 'data' / 'data'

def get_results_path():
    """Get results path - always in classification directory."""
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    return results_dir

def aggressive_cleanup():
    """Aggressive CPU and GPU memory cleanup to minimize RAM usage."""
    # Multiple garbage collection passes for CPU RAM optimization
    for _ in range(3):
        gc.collect()
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Print CPU RAM usage for monitoring
    ram_usage = psutil.virtual_memory()
    print(f"    CPU RAM: {ram_usage.percent:.1f}% ({ram_usage.used/1024**3:.1f}GB/{ram_usage.total/1024**3:.1f}GB)")

def save_model_result_immediately(model_name, best_params, best_score, strategy='normal'):
    """Save single model result immediately - simple version"""
    results_path = get_results_path()
    results_file = results_path / f"optimal_parameters_{strategy}_latest.json"
    
    # Load existing results or create new
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    # Update with new result
    results[model_name] = {
        'params': best_params,
        'score': best_score
    }
    
    # Save immediately
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  💾 Saved {model_name} to {results_file.name}")

# Model parameter grids - optimized for speed (always balanced class weights)
PARAM_GRIDS = {
    'eegnet': {
        'dropoutRate': [0.2, 0.3, 0.4],  # Lower dropout to prevent collapse
        'kernLength': [64, 128],         # Focus on proven architectures
        'F1': [8, 16],                   # Number of temporal filters
        'D': [2, 4],                     # Depth multiplier  
        'F2': [16, 32]                   # Number of pointwise filters
    },
    'cbramod': {
        'dropout': [0.1, 0.2],
        'hidden_dim': [256, 512],
        'label_smoothing': [0.0, 0.1]
    },
    'csp': {
        'n_components': [4, 6, 8, 10],
        'clf_type': ['lda', 'rf'],
        'reg': [0.01, 0.1, None]
    },
    'features_rf': {
        'n_estimators': [100, 200],
        'max_depth': [20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'features_svm': {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 0.01],
        'kernel': ['rbf', 'linear']
    },
    'et_rf': {
        'n_estimators': [100, 200],
        'max_depth': [20, None],
        'min_samples_split': [2, 5]
    },
    'et_svm': {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 0.01],
        'kernel': ['rbf', 'linear']
    }
}

# Data splitting strategies
DATA_SPLIT_STRATEGIES = {
    'normal': {'test_size': 0.2, 'val_size': 0.25},
    'leave_one_out': {'test_size': 0.15, 'val_size': 0.15}
}

def load_model_data(model_name, subjects, selected_channels=None, split_strategy='normal'):
    """Load and split data for a specific model with CPU RAM optimization"""
    data_loader = DataLoader(get_data_path())
    
    if model_name in ['eegnet', 'cbramod', 'csp']:
        if selected_channels is None:
            selected_channels = CBRAMOD_CHANNELS if model_name == 'cbramod' else DEFAULT_EEG_CHANNELS
        
        # Load EEG data with float32 for CPU RAM efficiency
        X, y = data_loader.load_labeled_eeg_data(subjects=subjects, 
                                                 channels=selected_channels,
                                                 dtype=np.float32)
    
    elif model_name.startswith('features_'):
        # Load advanced features with float32
        X, y = data_loader.load_features(subjects=subjects, 
                                         feature_type='advanced',
                                         dtype=np.float32)
    
    elif model_name.startswith('et_'):
        # Load ET features with float32
        X, y = data_loader.load_features(subjects=subjects, 
                                         feature_type='et',
                                         dtype=np.float32)
    
    # Print data info for CPU RAM monitoring
    print(f"  Data shape: {X.shape}, dtype: {X.dtype}")
    print(f"  Memory usage: {X.nbytes / 1024**2:.1f} MB")
    
    # Split data based on strategy
    split_config = DATA_SPLIT_STRATEGIES[split_strategy]
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=split_config['test_size'], 
        random_state=42, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=split_config['val_size'], 
        random_state=42, stratify=y_temp
    )
    
    # Cleanup temp variables to save CPU RAM
    del X_temp, y_temp
    gc.collect()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_model_instance(model_name, X_train, y_train, **params):
    """Create model instance with parameters and CPU RAM optimization"""
    
    if model_name == 'eegnet':
        # EEGNet always uses balanced class weights for binary classification
        class_weight = {0: 1.0, 1: len(y_train[y_train==0]) / len(y_train[y_train==1])}
        model = EEGNetClassifier(
            nb_classes=2,
            Chans=X_train.shape[2] if len(X_train.shape) > 2 else X_train.shape[1],
            Samples=X_train.shape[3] if len(X_train.shape) > 3 else X_train.shape[2],
            class_weight=class_weight,
            **params
        )
    
    elif model_name == 'cbramod':
        # CBraMod uses pos_weight for class imbalance
        pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        model = CBraModClassifier(
            input_shape=X_train.shape[1:],
            num_classes=2,
            pos_weight=pos_weight,
            **params
        )
    
    elif model_name == 'csp':
        # CSP always uses balanced class weights
        model = CSPClassifier(**params)
    
    elif model_name.startswith('features_'):
        # Feature classifiers always use balanced class weights
        model = FeatureClassifier(**params)
    
    elif model_name.startswith('et_'):
        # ET Feature classifiers always use balanced class weights
        model = ETFeatureClassifier(**params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def optimize_single_model(model_name, subjects, max_trials, split_strategy='normal', 
                         selected_channels=None, test_mode=False):
    """Optimize single model with CPU RAM optimization"""
    print(f"  Loading data for {model_name}...")
    
    # Load data with CPU RAM optimization
    X, y = load_model_data(model_name, subjects, selected_channels, split_strategy)[:2]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Clean up original data to save CPU RAM
    del X, y
    aggressive_cleanup()
    
    print(f"  Training set: {X_train.shape}, Validation: {X_val.shape}")
    print(f"  Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")
    
    param_grid = PARAM_GRIDS[model_name]
    
    # Limit parameter combinations if needed
    param_combinations = list(ParameterGrid(param_grid))
    if len(param_combinations) > max_trials:
        param_combinations = param_combinations[:max_trials]
    
    print(f"  Testing {len(param_combinations)} parameter combinations")
    
    best_score = -1
    best_params = None
    
    for i, params in enumerate(param_combinations):
        if test_mode and i >= 1:  # Only test first combination in test mode
            break
            
        try:
            print(f"    Trial {i+1}/{len(param_combinations)}: {params}")
            
            # Create model with CPU RAM optimization
            model = create_model_instance(model_name, X_train, y_train, **params)
            
            if test_mode:
                # Quick test - just create model
                score = 0.5
            else:
                # Train and evaluate with lower learning rate to prevent collapse
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
            
            print(f"      Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"      🎯 New best: {score:.4f}")
            
            # Cleanup model to save CPU RAM
            del model
            aggressive_cleanup()
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            aggressive_cleanup()
            continue
    
    print(f"  Best {model_name}: {best_score:.4f} with {best_params}")
    
    # Save results immediately 
    save_model_result_immediately(model_name, best_params, best_score, split_strategy)
    
    # Clean up data variables to free CPU RAM
    del X_train, X_val, y_train, y_val
    aggressive_cleanup()
    
    return best_params, best_score

def optimize_all_models(subjects=None, max_trials=15, split_strategies=['normal'], test_mode=False):
    """Optimize all models - saves results after each model"""
    if subjects is None:
        subjects = SUBJECTS[:5]
    
    results_path = get_results_path()
    
    print("=== MODEL OPTIMIZATION (CPU RAM OPTIMIZED) ===")
    print(f"Subjects: {subjects}")
    print(f"Max trials per model: {max_trials}")
    print(f"Results saved after each model (no data loss)")
    
    all_results = {}
    
    for strategy in split_strategies:
        print(f"\n{'='*50}")
        print(f"SPLIT STRATEGY: {strategy.upper()}")
        print(f"{'='*50}")
        
        optimal_params = {}
        
        for i, model_name in enumerate(PARAM_GRIDS.keys(), 1):
            print(f"\n🔄 Optimizing {model_name} ({i}/{len(PARAM_GRIDS)})...")
            try:
                best_params, best_score = optimize_single_model(
                    model_name, subjects, max_trials, strategy, test_mode=test_mode
                )
                optimal_params[model_name] = {
                    'params': best_params,
                    'score': best_score
                }
                
                print(f"✅ {model_name}: {best_score:.4f}")
                
                # Extra cleanup after each model
                aggressive_cleanup()
                    
            except Exception as e:
                print(f"❌ ERROR with {model_name}: {e}")
                aggressive_cleanup()
                continue
        
        all_results[strategy] = optimal_params
    
    # Save timestamped final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for strategy, optimal_params in all_results.items():
        strategy_file = results_path / f"optimal_parameters_{strategy}_{timestamp}.json"
        
        with open(strategy_file, 'w') as f:
            json.dump(optimal_params, f, indent=2, default=str)
        
        print(f"\n💾 Final {strategy} results: {strategy_file}")
    
    # Final cleanup
    aggressive_cleanup()
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    for strategy, optimal_params in all_results.items():
        print(f"\n{strategy.upper()} Strategy:")
        for model_name, result in optimal_params.items():
            print(f"  {model_name:12}: {result['score']:.4f}")
    
    return all_results

# Simple runner functions
def quick_optimization():
    """Quick optimization with few subjects"""
    subjects = SUBJECTS[:3]
    return optimize_all_models(subjects, max_trials=8, split_strategies=['normal'])

def full_optimization():
    """Full optimization with all subjects"""
    return optimize_all_models(SUBJECTS, max_trials=20, split_strategies=['normal'])

def test_all_models():
    """Test model creation and data loading without training (for debugging)"""
    subjects = ['P10', 'P16']
    return optimize_all_models(subjects, max_trials=1, split_strategies=['normal'], test_mode=True)

if __name__ == "__main__":
    # Run quick optimization by default
    print("Starting quick optimization...")
    quick_optimization()