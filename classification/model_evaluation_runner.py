"""
Model Evaluation Runner

Evaluates trained models using optimal parameters from optimization phase.
Separate from optimization to keep notebooks clean and focused.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

# Import models
from classification.models.eeg_net.eegnet_classifier import EEGNetClassifier
from classification.models.cbramod.cbramod_classifier import CBraModClassifier
from classification.models.csp.csp_classifier import CSPClassifier
from classification.models.features.feature_classifier import FeatureClassifier
from classification.models.features.et_feature_classifier import ETFeatureClassifier

# Import from optimization runner
from classification.model_optimization_runner import (
    get_data_path, get_results_path, load_model_data, aggressive_cleanup
)
from data.data_configs import SUBJECTS

def evaluate_single_model(model_name, params, subjects, selected_channels=None):
    """Evaluate single model with given parameters"""
    
    # Load data with train/test split
    X, y = load_model_data(model_name, subjects, selected_channels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create model with optimal parameters (always with balanced class weights)
    if model_name == 'eegnet':
        model = EEGNetClassifier(**params)
    elif model_name == 'cbramod':
        model = CBraModClassifier(**params)
    elif model_name == 'csp':
        # Always use balanced class weights for RF
        if 'clf_type' in params and params['clf_type'] == 'rf':
            params_with_weights = {**params, 'class_weight': 'balanced'}
        else:
            params_with_weights = params
        model = CSPClassifier(**params_with_weights)
    elif model_name == 'features_rf':
        # Always use balanced class weights
        rf_params = {**params, 'class_weight': 'balanced'}
        model = FeatureClassifier(model_type='rf', **rf_params)
    elif model_name == 'features_svm':
        # Always use balanced class weights
        svm_params = {**params, 'class_weight': 'balanced'}
        model = FeatureClassifier(model_type='svm', **svm_params)
    elif model_name == 'et_rf':
        # Always use balanced class weights
        rf_params = {**params, 'class_weight': 'balanced'}
        model = ETFeatureClassifier(model_type='rf', **rf_params)
    elif model_name == 'et_svm':
        # Always use balanced class weights
        svm_params = {**params, 'class_weight': 'balanced'}
        model = ETFeatureClassifier(model_type='svm', **svm_params)
    
    # Train model - use a small validation set for training monitoring
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    try:
        results = model.train(X_train_split, y_train_split, X_val_split, y_val_split, X_test, y_test)
        
        # Extract test results
        test_results = {
            'accuracy': results.get('score', results['balanced_acc']),
            'balanced_accuracy': results['balanced_acc'], 
            'f1_macro': results.get('f1_macro', 0.0)
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
    
    # Clean up
    del model, X, y, X_train, X_test, y_train, y_test
    aggressive_cleanup()
    
    return test_results

def evaluate_all_models(subjects=None, selected_channels=None, split_strategy='normal'):
    """Train and evaluate all models using optimal parameters"""
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
    print(f"Results saved after each model (no data loss)")
    
    # Load existing evaluation results or start fresh
    eval_file = results_path / f"evaluation_results_{split_strategy}_latest.json"
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            evaluation_results = json.load(f)
        print(f"📋 Loaded existing results: {len(evaluation_results)} models")
    else:
        evaluation_results = {}
    
    for i, (model_name, param_info) in enumerate(optimal_params.items(), 1):
        if model_name in evaluation_results:
            print(f"⏭️  Skipping {model_name} (already evaluated)")
            continue
            
        try:
            print(f"\\n🔄 Evaluating {model_name} ({i}/{len(optimal_params)})...")
            params = param_info['params']
            
            test_results = evaluate_single_model(model_name, params, subjects, selected_channels)
            evaluation_results[model_name] = test_results
            
            print(f"  Accuracy: {test_results['accuracy']:.4f}")
            print(f"  Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
            print(f"  F1-Macro: {test_results['f1_macro']:.4f}")
            
            # Save immediately after each model
            with open(eval_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            print(f"  💾 Saved to {eval_file.name}")
            
            # Cleanup after each model to save CPU RAM
            aggressive_cleanup()
            
        except Exception as e:
            print(f"❌ ERROR evaluating {model_name}: {e}")
            aggressive_cleanup()
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
    
    # Final memory cleanup
    aggressive_cleanup()
    
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
def quick_evaluation(split_strategy='normal'):
    """Quick evaluation with few subjects"""
    subjects = SUBJECTS[:3]
    return evaluate_all_models(subjects, split_strategy=split_strategy)

def full_evaluation(split_strategy='normal'):
    """Full evaluation with all subjects"""
    return evaluate_all_models(SUBJECTS, split_strategy=split_strategy)