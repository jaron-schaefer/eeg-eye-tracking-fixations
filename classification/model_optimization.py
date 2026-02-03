"""
Model Optimization Pipeline

Three functions:
1. optimize_all_models() - Find best parameters for each model
2. evaluate_all_models() - Train and test with optimal parameters  
3. test_channel_configurations() - Test different channel groups
"""

import numpy as np
import pandas as pd
import json
import pickle
import gc
import torch
from pathlib import Path
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

# Import models
from classification.models.eeg_net.eegnet_classifier import EEGNetClassifier
from classification.models.cbramod.cbramod_classifier import CBraModClassifier
from classification.models.csp.csp_classifier import CSPClassifier
from classification.models.features.feature_classifier import FeatureClassifier
from classification.models.features.et_feature_classifier import ETFeatureClassifier

# Import data utilities
from data.data_loader import DataLoader
from data.data_configs import CBRAMOD_CHANNELS, SUBJECTS
import os

# Path configuration for different environments
def get_data_path():
    """Get data path - for Colab, data is next to repository folder."""
    current_dir = Path(__file__).parent.absolute()
    if '/content/' in str(current_dir):  # Running in Colab
        return Path('/content/eye-eeg')
    else:  # Running locally
        return current_dir.parent / 'data'

def get_results_path():
    """Get results path - always in classification directory."""
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    return results_dir

# EEG Channel mappings (names to indices)
DEFAULT_EEG_CHANNELS = ['F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6',
                        'Fp1', 'Fp2', 'Pz', 'P3', 'P4', 'P7', 'P8']

# Model parameter grids - reduced for efficiency but educated choices
PARAM_GRIDS = {
    'eegnet': {
        'dropoutRate': [0.3, 0.5],  # Most common effective values
        'kernLength': [64, 128],  # Focus on proven architectures
        'F1': [8, 16],   # Number of temporal filters
        'D': [2, 4],     # Depth multiplier
        'F2': [16, 32]   # Number of pointwise filters
    },
    'cbramod': {
        'dropout': [0.1, 0.2],  # Most effective range for transformers
        'hidden_dim': [256, 512],  # Focus on medium-large architectures
        'label_smoothing': [0.0, 0.1]  # Label smoothing options
    },
    'csp': {
        'n_components': [4, 6, 8, 10],  # Reduced but covers key range
        'clf_type': ['lda', 'rf'],  # Only supported types
        'reg': [0.01, 0.1, None]  # Simplified regularization (correct parameter name)
    },
    'features_rf': {
        'n_estimators': [100, 200],  # Focus on proven values
        'max_depth': [20, None],     # Most effective options
        'min_samples_split': [2, 5], # Key values
        'min_samples_leaf': [1, 2]   # Essential options
    },
    'features_svm': {
        'C': [0.1, 1.0, 10.0],      # Core C values
        'gamma': ['scale', 0.01],    # Most effective options
        'kernel': ['rbf', 'linear']   # Key kernels
    },
    'et_rf': {
        'n_estimators': [100, 200],  # Proven values
        'max_depth': [20, None],     # Most effective
        'min_samples_split': [2, 5]  # Key options
    },
    'et_svm': {
        'C': [0.1, 1.0, 10.0],      # Core values
        'gamma': ['scale', 0.01],    # Most effective
        'kernel': ['rbf', 'linear']   # Key kernels
    }
}

# Data splitting strategies
DATA_SPLIT_STRATEGIES = {
    'normal': {'test_size': 0.2, 'val_size': 0.25},
    'leave_one_out': {'test_size': 0.15, 'val_size': 0.15}  # Smaller splits for more training data
}

# Global data cache to avoid reloading within single optimization run
_DATA_CACHE = {}

def clear_data_cache():
    """Clear the global data cache and force garbage collection."""
    global _DATA_CACHE
    _DATA_CACHE.clear()
    # Force garbage collection
    gc.collect()
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Data cache cleared and memory freed.")

def cleanup_memory():
    """Clean up memory after each model evaluation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model_data(model_name, subjects, selected_channels=None, use_cache=True):
    """Load data for specific model type with caching for RAM efficiency
    """
    
    # Create cache key
    cache_key = f"{model_name}_{'-'.join(subjects)}_{selected_channels}"
    
    if use_cache and cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]
    
    # Get data path (adjusts for Colab environment)
    data_path = get_data_path()
    
    if model_name in ['eegnet', 'cbramod', 'csp']:
        # Raw EEG data using load_labeled_eeg_data method
        data_loader = DataLoader(data_path=str(data_path))
        
        # Use default channels if none provided for raw EEG models
        if selected_channels is None:
            selected_channels = DEFAULT_EEG_CHANNELS
            
        # Use the correct path for labeled EEG data
        labeled_eeg_dir = data_path / 'data' / 'labeled_eeg_data'
        X, y, _ = data_loader.load_labeled_eeg_data(subjects, selected_channels, labeled_eeg_data_dir=labeled_eeg_dir)
        
    elif model_name.startswith('features_'):
        # EEG features with channel filtering support
        data_loader = DataLoader(data_path=str(data_path))
        # selected_channels should already be channel names (strings)
        channel_names = selected_channels  # No conversion needed
        
        X, y, _, _, _ = data_loader.load_features(subjects, mode="advanced", channels=channel_names)
        
    elif model_name.startswith('et_'):
        # ET features using load_et_features instance method
        data_loader = DataLoader(data_path=str(data_path))
        X, y, _, _, _ = data_loader.load_et_features(subjects)
        
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Cache the data if using cache
    if use_cache:
        _DATA_CACHE[cache_key] = (X, y)
        
    return X, y

def optimize_single_model(model_name, subjects, max_trials=10, split_strategy='normal', test_mode=False):
    """Optimize parameters for one model
    
    Args:
        model_name: Name of the model to optimize
        subjects: List of subject IDs
        max_trials: Maximum number of parameter combinations to try
        split_strategy: Data split strategy to use
        test_mode: If True, only test model creation without training
    """
    print(f"Optimizing {model_name} with {split_strategy} split...")
    
    # Load data once
    X, y = load_model_data(model_name, subjects, CBRAMOD_CHANNELS)
    
    # Get split sizes based on strategy
    splits = DATA_SPLIT_STRATEGIES[split_strategy]
    test_size = splits['test_size']
    val_size = splits['val_size']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size + val_size, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=val_size/(test_size + val_size), random_state=42, stratify=y_val
    )
    
    # Get parameter combinations
    param_combinations = list(ParameterGrid(PARAM_GRIDS[model_name]))
    if len(param_combinations) > max_trials:
        np.random.shuffle(param_combinations)
        param_combinations = param_combinations[:max_trials]
    
    best_score = -1
    best_params = None
    
    for params in param_combinations:
        try:
            # Create model
            if model_name == 'eegnet':
                model = EEGNetClassifier(**params)
            elif model_name == 'cbramod':
                model = CBraModClassifier(**params)
            elif model_name == 'csp':
                model = CSPClassifier(**params)
            elif model_name == 'features_rf':
                model = FeatureClassifier(model_type='rf', **{k:v for k,v in params.items() if k != 'model_type'})
            elif model_name == 'features_svm':
                model = FeatureClassifier(model_type='svm', **{k:v for k,v in params.items() if k != 'model_type'})
            elif model_name == 'et_rf':
                model = ETFeatureClassifier(model_type='rf', **{k:v for k,v in params.items() if k != 'model_type'})
            elif model_name == 'et_svm':
                model = ETFeatureClassifier(model_type='svm', **{k:v for k,v in params.items() if k != 'model_type'})
            
            if test_mode:
                # Just test model creation and data shape compatibility
                print(f"  ✓ Model created successfully with {params}")
                print(f"  ✓ Data shape: {X_train.shape}, Labels: {y_train.shape}")
                score = 0.5  # Dummy score for test mode
            else:
                # Train and evaluate - free memory after each trial
                results = model.train(X_train, y_train, X_val, y_val)
                
                if 'validation' in results:
                    score = results['validation']['balanced_accuracy']
                else:
                    score = results['test']['balanced_accuracy']
                
            if score > best_score:
                best_score = score
                best_params = params.copy()
                
            # Clean up memory after each trial
            del model
            if not test_mode:
                del results
            cleanup_memory()
                
        except Exception as e:
            print(f"  Failed with {params}: {e}")
            cleanup_memory()  # Clean up even on failure
            continue
    
    print(f"  Best {model_name}: {best_score:.4f} with {best_params}")
    return best_params, best_score

def optimize_all_models(subjects=None, max_trials=15, split_strategies=['normal', 'leave_one_out'], test_mode=False):
    """
    Find optimal parameters for all models and save to file
    
    Args:
        subjects: List of subject IDs
        max_trials: Maximum number of parameter combinations per model
        split_strategies: List of data split strategies
        test_mode: If True, only test model creation without training
    """
    if subjects is None:
        subjects = SUBJECTS[:5]  # Use subset for optimization
    
    # Create results directory in classification folder
    results_path = get_results_path()
    
    print("=== OPTIMIZING ALL MODELS ===")
    print(f"Subjects: {subjects}")
    print(f"Max trials per model: {max_trials}")
    print(f"Split strategies: {split_strategies}")
    print(f"Test mode: {test_mode}")
    print(f"Results will be saved to: {results_path}")
    
    all_results = {}
    
    for strategy in split_strategies:
        print(f"\\n{'='*50}")
        print(f"SPLIT STRATEGY: {strategy.upper()}")
        print(f"{'='*50}")
        
        optimal_params = {}
        
        for model_name in PARAM_GRIDS.keys():
            try:
                best_params, best_score = optimize_single_model(
                    model_name, subjects, max_trials, strategy, test_mode
                )
                optimal_params[model_name] = {
                    'params': best_params,
                    'score': best_score
                }
                
                # Clear cache periodically to manage RAM
                if len(optimal_params) % 3 == 0:
                    clear_data_cache()
                    
            except Exception as e:
                print(f"ERROR with {model_name}: {e}")
                continue
        
        all_results[strategy] = optimal_params
    
    # Save results with timestamp for both strategies
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for strategy, optimal_params in all_results.items():
        strategy_file = results_path / f"optimal_parameters_{strategy}_{timestamp}.json"
        strategy_latest = results_path / f"optimal_parameters_{strategy}_latest.json"
        
        # Save timestamped version
        with open(strategy_file, 'w') as f:
            json.dump(optimal_params, f, indent=2, default=str)
        
        # Save latest version
        with open(strategy_latest, 'w') as f:
            json.dump(optimal_params, f, indent=2, default=str)
        
        print(f"\\n{strategy.upper()} strategy results saved to: {strategy_file}")
    
    # Also save combined results
    combined_file = results_path / f"optimal_parameters_all_{timestamp}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Clear cache at the end
    clear_data_cache()
    
    # Print summary for all strategies
    print(f"\\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    for strategy, optimal_params in all_results.items():
        print(f"\\n{strategy.upper()} Strategy:")
        for model_name, result in optimal_params.items():
            print(f"  {model_name:12}: {result['score']:.4f}")
    
    return all_results

def evaluate_single_model(model_name, params, subjects, selected_channels=None):
    """Evaluate single model with given parameters
    
    Args:
        model_name: Name of the model to evaluate
        params: Model parameters
        subjects: List of subject IDs
        selected_channels: List of channel names (strings) to use for features
    """
    
    # Load data with train/test split
    X, y = load_model_data(model_name, subjects, selected_channels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create model with optimal parameters
    if model_name == 'eegnet':
        model = EEGNetClassifier(**params)
    elif model_name == 'cbramod':
        model = CBraModClassifier(**params)
    elif model_name == 'csp':
        model = CSPClassifier(**params)
    elif model_name == 'features_rf':
        model = FeatureClassifier(model_type='rf', **params)
    elif model_name == 'features_svm':
        model = FeatureClassifier(model_type='svm', **params)
    elif model_name == 'et_rf':
        model = ETFeatureClassifier(model_type='rf', **params)
    elif model_name == 'et_svm':
        model = ETFeatureClassifier(model_type='svm', **params)
    
    # Train model - use a small validation set for training monitoring
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    try:
        results = model.train(X_train_split, y_train_split, X_val_split, y_val_split, X_test, y_test)
        
        # Extract test results - try different result structures
        if 'test' in results:
            test_results = results['test']
        elif 'validation' in results:
            # If no test results, use validation and manually evaluate on test
            y_pred = model.predict(X_test)
            test_results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro')
            }
        else:
            # Fallback to manual evaluation
            y_pred = model.predict(X_test)
            test_results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro')
            }
            
    except Exception as e:
        print(f"  Error in model training: {e}")
        # Fallback: train without validation monitoring
        model.fit(X_train, y_train) if hasattr(model, 'fit') else None
        y_pred = model.predict(X_test)
        test_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        }
    
    return test_results

def evaluate_all_models(subjects=None, selected_channels=None, split_strategy='normal'):
    """
    Train and evaluate all models using optimal parameters
    """
    if subjects is None:
        subjects = SUBJECTS
    
    results_path = get_results_path()

    
    # Load optimal parameters for the specified strategy
    param_file = results_path / f"optimal_parameters_{split_strategy}_latest.json"
    
    if not Path(param_file).exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}. Run optimize_all_models() first.")
    
    with open(param_file, 'r') as f:
        optimal_params = json.load(f)
    
    print(f"=== EVALUATING ALL MODELS ({split_strategy.upper()} SPLIT) ===")
    print(f"Subjects: {subjects}")
    print(f"Using parameters from: {param_file}")
    
    evaluation_results = {}
    
    for model_name, param_info in optimal_params.items():
        try:
            print(f"\\nEvaluating {model_name}...")
            params = param_info['params']
            
            test_results = evaluate_single_model(model_name, params, subjects, selected_channels)
            evaluation_results[model_name] = test_results
            
            print(f"  Accuracy: {test_results['accuracy']:.4f}")
            print(f"  Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
            print(f"  F1-Macro: {test_results['f1_macro']:.4f}")
            
        except Exception as e:
            print(f"ERROR evaluating {model_name}: {e}")
            continue
    
    # Save evaluation results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f"evaluation_results_{split_strategy}_{timestamp}.json"
    latest_results = results_path / f"evaluation_results_{split_strategy}_latest.json"
    
    # Save timestamped and latest versions
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
        
    with open(latest_results, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"\\nEvaluation results saved to: {results_file}")
    print(f"Latest results saved to: {latest_results}")
    
    # Clear cache
    clear_data_cache()
    
    # Print final summary
    print("\\n=== FINAL RESULTS ===")
    results_df = pd.DataFrame(evaluation_results).T
    results_df = results_df.sort_values('balanced_accuracy', ascending=False)
    print(results_df.round(4))
    
    best_model = results_df.index[0]
    best_score = results_df.iloc[0]['balanced_accuracy']
    print(f"\\nBest Model: {best_model} (Balanced Accuracy: {best_score:.4f})")
    
    return evaluation_results



# Simple runner functions
def quick_optimization():
    """Quick optimization with few subjects"""
    subjects = SUBJECTS[:3]  # Quick test
    return optimize_all_models(subjects, max_trials=8, split_strategies=['normal'])

def full_optimization():
    """Full optimization with all subjects and both split strategies"""
    return optimize_all_models(SUBJECTS, max_trials=20, split_strategies=['normal', 'leave_one_out'])

def quick_evaluation(split_strategy='normal'):
    """Quick evaluation with few subjects"""
    subjects = SUBJECTS[:3]
    return evaluate_all_models(subjects, split_strategy=split_strategy)

def full_evaluation(split_strategy='normal'):
    """Full evaluation with all subjects"""
    return evaluate_all_models(SUBJECTS, split_strategy=split_strategy)

def test_all_models():
    """Test model creation and data loading without training (for debugging)"""
    # Use subjects that actually have data
    subjects = ['P10', 'P16']  # These were found in the data directory
    return optimize_all_models(subjects, max_trials=1, split_strategies=['normal'], test_mode=True)

def test_single_model_type(model_name):
    """Test a specific model type"""
    subjects = SUBJECTS[:2]
    print(f"Testing {model_name}...")
    try:
        X, y = load_model_data(model_name, subjects, CBRAMOD_CHANNELS)
        print(f"✓ Data loaded: {X.shape}, {y.shape}")
        
        # Test one parameter combination
        params = list(ParameterGrid(PARAM_GRIDS[model_name]))[0]
        print(f"✓ Testing parameters: {params}")
        
        if model_name == 'eegnet':
            model = EEGNetClassifier(**params)
        elif model_name == 'cbramod':
            model = CBraModClassifier(**params)
        elif model_name == 'csp':
            model = CSPClassifier(**params)
        elif model_name == 'features_rf':
            model = FeatureClassifier(model_type='rf', **params)
        elif model_name == 'features_svm':
            model = FeatureClassifier(model_type='svm', **params)
        elif model_name == 'et_rf':
            model = ETFeatureClassifier(model_type='rf', **params)
        elif model_name == 'et_svm':
            model = ETFeatureClassifier(model_type='svm', **params)
            
        print(f"✓ Model created: {model}")
        return True
        
    except Exception as e:
        print(f"✗ Error testing {model_name}: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    print("EEG Model Optimization Pipeline")
    print("1. Run optimize_all_models() to find best parameters")
    print("2. Run evaluate_all_models() to test with optimal parameters")
    
    # Uncomment to run:
    # optimize_all_models()
    # evaluate_all_models()