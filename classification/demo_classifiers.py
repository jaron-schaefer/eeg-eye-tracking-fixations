"""
Demo script showing how to use the integrated classifiers
"""

import numpy as np
from data.data_loader import load_training_data
from classification.models.csp.csp_classifier import CSPClassifier
from classification.models.eeg_net.eegnet_classifier import EEGNetClassifier
from classification.models.cbramod.cbramod_classifier import CBraModClassifier

def demo_classifiers():
    """Demo all three classifiers"""
    print("🧠 EEG Classification Demo")
    print("=" * 50)
    
    # Load data (replace with your data loading)
    print("Loading data...")
    try:
        # Example data loading - adjust for your data
        X_train, y_train, X_val, y_val, X_test, y_test, subject_ids = load_training_data()
        print(f"Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test samples")
    except:
        print("❌ Could not load data. Please implement your data loading logic.")
        return
    
    # 1. CSP Classifier
    print("\n" + "="*20 + " CSP " + "="*20)
    csp = CSPClassifier()
    csp_results = csp.train(X_train, y_train, X_val, y_val, X_test, y_test, subject_ids=subject_ids)
    
    # 2. EEGNet Classifier  
    print("\n" + "="*18 + " EEGNet " + "="*18)
    eegnet = EEGNetClassifier()
    eegnet_results = eegnet.train(X_train, y_train, X_val, y_val, X_test, y_test, subject_ids=subject_ids)
    
    # 3. CBraMod Classifier
    print("\n" + "="*17 + " CBraMod " + "="*17)
    cbramod = CBraModClassifier()
    cbramod_results = cbramod.train(X_train, y_train, X_val, y_val, X_test, y_test, subject_ids=subject_ids)
    
    # Summary
    print("\n" + "="*20 + " SUMMARY " + "="*20)
    print(f"CSP:     {csp_results['balanced_acc']:.4f}")
    print(f"EEGNet:  {eegnet_results['balanced_acc']:.4f}") 
    print(f"CBraMod: {cbramod_results['balanced_acc']:.4f}")

def demo_optimization():
    """Demo parameter optimization"""
    print("\n🔧 Parameter Optimization Demo")
    print("=" * 50)
    
    # Load data
    X_train, y_train, X_val, y_val, _, _, subject_ids = load_training_data()
    
    # Optimize CSP
    print("\nOptimizing CSP...")
    csp = CSPClassifier()
    csp_best = csp.optimize(X_train, y_train, X_val, y_val, subject_ids=subject_ids)
    print(f"Best CSP params: {csp_best['best_params']}")
    
    # You can do the same for EEGNet and CBraMod
    # eegnet = EEGNetClassifier()
    # eegnet_best = eegnet.optimize(X_train, y_train, X_val, y_val, subject_ids=subject_ids)

if __name__ == "__main__":
    demo_classifiers()
    # demo_optimization()