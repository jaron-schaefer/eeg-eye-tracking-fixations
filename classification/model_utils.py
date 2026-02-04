import gc
import numpy as np
import pandas as pd
import json
from pathlib import Path
import psutil
from sklearn.model_selection import ParameterGrid
import torch

# Import models
from classification.models.eeg_net.eegnet_classifier import EEGNetClassifier
from classification.models.cbramod.cbramod_classifier import CBraModClassifier
from classification.models.csp.csp_classifier import CSPClassifier
from classification.models.features.feature_classifier import FeatureClassifier
from classification.models.features.et_feature_classifier import ETFeatureClassifier
from data.data_configs import SUBJECTS

# Import data utilities
from data.data_loader import DataLoader
from data.data_configs import CBRAMOD_CHANNELS, SUBJECTS, DEFAULT_EEG_CHANNELS
from classification.training_split import split_normal, split_leave_n_out_generator, split_leave_n_out_generator, split_leave_n_out_generator

def cleanup():
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


def get_model(model_name: str, params: dict):
    """Instantiate model based on name and parameters"""
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def load_model_data(model_name, subjects, selected_channels=None, split_strategy='normal'):
    """Load and split data for a specific model with CPU RAM optimization (for optimization use)"""
    # Handle DataLoader path configuration based on environment
    current_dir = Path(__file__).parent.absolute()
    
    if '/content/' in str(current_dir):  # Running in Colab
        # For Colab, set the data path to drive location
        DataLoader.data_path = Path('/content/drive/MyDrive/eye-eeg/data')
    # For local, use default paths (no change needed)
    
    if selected_channels is None:
        selected_channels = CBRAMOD_CHANNELS

    if model_name in ['eegnet', 'cbramod', 'csp']:
        X, y, subject_ids, stimulus_summary = DataLoader.load_labeled_eeg_data(
            subjects=subjects, 
            channels=selected_channels
        )
        X = X.astype(np.float32)  # Convert after loading
    
    elif model_name.startswith('features_'):
        X, y, feature_names, subject_ids, stimulus_summary = DataLoader.load_features(
            subjects=subjects,
            mode='advanced',
            channels=selected_channels
        )
        X = X.astype(np.float32)
    
    elif model_name.startswith('et_'):
        X, y, feature_names, subject_ids, stimulus_summary = DataLoader.load_et_features(
            subjects=subjects
        )
        X = X.astype(np.float32)
    
    # Print data info for CPU RAM monitoring
    print(f"  Data shape: {X.shape}, dtype: {X.dtype}")
    print(f"  Memory usage: {X.nbytes / 1024**2:.1f} MB")
    
    # Split data using the appropriate split function based on strategy
    
    if split_strategy == 'leave_one_out':

        # Return the generator instead of a single split
        fold_generator = split_leave_n_out_generator(
            X, y,
            subjects=subject_ids,
            # stimulus_summary=stimulus_summary,
        )

        X_train, X_val, X_test, y_train, y_val, y_test = next(fold_generator)[:6]
    else:
        # Use split_normal function with stimulus summary support
        result = split_normal(
            X, y,
            # stimulus_summary=stimulus_summary,
        )
        
        # Extract the basic splits (ignore stimulus data for this function)
        X_train, X_val, X_test, y_train, y_val, y_test = result[:6]
        
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_single_model(model_name, subjects, params, selected_channels=None):
    
    # Load data with train/test split
    X_train, X_val, X_test, y_train, y_val, y_test = load_model_data(model_name, subjects, selected_channels)
    
    # Combine train and val for evaluation (since we're not optimizing)
    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    
    # Create model with optimal parameters (always with balanced class weights)
    model = get_model(model_name, params)
    results = model.train(X, y, X_test, y_test)

    del model, X, y, X_train, X_test, y_train, y_test
    cleanup()
    
    return results

def optimize_single_model(model_name, subjects, param_grid, max_trials=15, split_strategy='normal', 
                         selected_channels=None):
    """Optimize single model with CPU RAM optimization"""

    X_train, X_val, X_test, y_train, y_val, y_test = load_model_data(model_name, subjects, selected_channels)

    # Clean up test data since we don't need it for optimization
    del X_test, y_test
    cleanup()

    # Limit parameter combinations if needed
    param_combinations = list(ParameterGrid(param_grid))
    if len(param_combinations) > max_trials:
        param_combinations = param_combinations[:max_trials]
        
    best_score = -1
    best_params = None
    
    for i, params in enumerate(param_combinations):
        
        model = get_model(model_name, params)
        results = model.train(X_train, y_train, X_val, y_val)
        score = results['balanced_acc']
            
        if score > best_score:
            best_score = score
            best_params = params        
    
    print(f"  Best {model_name}: {best_score:.4f} with {best_params}")
    
    # Save results immediately 
    save_model_result(model_name, best_params, best_score, split_strategy)
    
    # Clean up data variables to free CPU RAM
    del X_train, X_val, y_train, y_val
    
    return best_params, best_score


def save_model_result(model_name, best_params, best_score, split_strategy='normal'):

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"optimal_parameters_{split_strategy}_latest.json"
    
    # Load existing results or create new
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    # Check if we should update the result
    should_save = True
    if model_name in results:
        old_score = results[model_name].get('balanced_accuracy', results[model_name].get('score', 0))
        if best_score <= old_score:
            should_save = False

    if should_save:
        # Update with new result
        # Save both legacy 'score' key and explicit 'balanced_accuracy'
        results[model_name] = {
            'params': best_params,
            'score': best_score,
            'balanced_accuracy': best_score
        }
        
        # Save immediately
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        try:
            print(f"  💾 Saved {model_name} to {results_file.name} (balanced_accuracy={best_score:.4f})")
        except Exception:
            print(f"  💾 Saved {model_name} to {results_file.name}")


def optimize_all_models(param_grid, subjects, max_trials=15, split_strategies=['normal'], test_mode=False):

    for model in param_grid.keys():
        print(f"Optimizing {model}...")
        optimize_single_model(model, subjects, param_grid[model], max_trials=max_trials, split_strategy=split_strategies[0])


if __name__ == "__main__":
    params = {
    'n_estimators': [200],
    'max_depth': [30],
    'min_samples_split': [2],
    'min_samples_leaf': [1] 
}
    optimize_single_model("features_rf", SUBJECTS[:1], params, max_trials=2)










# def evaluate_all_models(subjects=None, selected_channels=None, split_strategy='normal'):
#     """Train and evaluate all models using optimal parameters"""
#     if subjects is None:
#         subjects = SUBJECTS
    
#     results_path = get_results_path()
    
#     # Load optimal parameters for the specified strategy
#     param_file = results_path / f"optimal_parameters_{split_strategy}_latest.json"
    
#     if not Path(param_file).exists():
#         raise FileNotFoundError(f"Parameter file not found: {param_file}. Run optimize_all_models() first.")
    
#     with open(param_file, 'r') as f:
#         optimal_params = json.load(f)
    
#     print(f"=== EVALUATING ALL MODELS ({split_strategy.upper()} SPLIT) ===")
#     print(f"Subjects: {subjects}")
#     print(f"Results saved after each model (no data loss)")
    
#     # Load existing evaluation results or start fresh
#     eval_file = results_path / f"evaluation_results_{split_strategy}_latest.json"
#     if eval_file.exists():
#         with open(eval_file, 'r') as f:
#             evaluation_results = json.load(f)
#         print(f"📋 Loaded existing results: {len(evaluation_results)} models")
#     else:
#         evaluation_results = {}
    
#     for i, (model_name, param_info) in enumerate(optimal_params.items(), 1):
#         if model_name in evaluation_results:
#             print(f"⏭️  Skipping {model_name} (already evaluated)")
#             continue
            
#         try:
#             print(f"\\n🔄 Evaluating {model_name} ({i}/{len(optimal_params)})...")
#             params = param_info['params']
            
#             test_results = evaluate_single_model(model_name, params, subjects, selected_channels)
#             evaluation_results[model_name] = test_results
            
#             print(f"  Accuracy: {test_results['accuracy']:.4f}")
#             print(f"  Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
#             print(f"  F1-Macro: {test_results['f1_macro']:.4f}")
            
#             # Save immediately after each model
#             with open(eval_file, 'w') as f:
#                 json.dump(evaluation_results, f, indent=2, default=str)
#             print(f"  💾 Saved to {eval_file.name}")
            
#             # Cleanup after each model to save CPU RAM
#             aggressive_cleanup()
            
#         except Exception as e:
#             print(f"❌ ERROR evaluating {model_name}: {e}")
#             aggressive_cleanup()
#             continue
    
#     # Save evaluation results with timestamp
#     from datetime import datetime
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_file = results_path / f"evaluation_results_{split_strategy}_{timestamp}.json"
#     latest_results = results_path / f"evaluation_results_{split_strategy}_latest.json"
    
#     # Save timestamped and latest versions
#     with open(results_file, 'w') as f:
#         json.dump(evaluation_results, f, indent=2, default=str)
        
#     with open(latest_results, 'w') as f:
#         json.dump(evaluation_results, f, indent=2, default=str)
    
#     print(f"\\nEvaluation results saved to: {results_file}")
#     print(f"Latest results saved to: {latest_results}")
    
#     # Final memory cleanup
#     aggressive_cleanup()
    
#     # Print final summary
#     print("\\n=== FINAL RESULTS ===")
#     results_df = pd.DataFrame(evaluation_results).T
#     results_df = results_df.sort_values('balanced_accuracy', ascending=False)
#     print(results_df.round(4))
    
#     best_model = results_df.index[0]
#     best_score = results_df.iloc[0]['balanced_accuracy']
#     print(f"\\nBest Model: {best_model} (Balanced Accuracy: {best_score:.4f})")
    
#     return evaluation_results