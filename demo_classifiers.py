"""
Demo script showing the new unified classifier architecture

This demonstrates how to use the new class-based approach for all EEG classifiers.
"""

import numpy as np
from classification.classifiers import CSPClassifier, EEGNetClassifier, CBraModClassifier, compare_classifiers

def create_demo_data():
    """Create some demo EEG data for testing"""
    print("Creating synthetic EEG data for demo...")
    
    # Simulate EEG data: (samples, channels, time_points)
    n_samples = 100
    n_channels = 19
    n_timepoints = 500
    
    # Create synthetic data with some patterns
    np.random.seed(42)
    X = np.random.randn(n_samples, n_channels, n_timepoints) * 10  # Scaled for CBraMod
    
    # Create labels with some pattern
    y = (X[:, :5, 100:200].mean(axis=(1,2)) > 0).astype(int)  # Based on first 5 channels
    
    # Split data
    split1, split2 = int(0.6 * n_samples), int(0.8 * n_samples)
    
    X_train, y_train = X[:split1], y[:split1]
    X_val, y_val = X[split1:split2], y[split1:split2]  
    X_test, y_test = X[split2:], y[split2:]
    
    print(f"  Train: {X_train.shape}, labels: {np.bincount(y_train)}")
    print(f"  Val:   {X_val.shape}, labels: {np.bincount(y_val)}")
    print(f"  Test:  {X_test.shape}, labels: {np.bincount(y_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def demo_individual_classifiers():
    """Demonstrate using individual classifiers"""
    print("\\n" + "="*70)
    print("INDIVIDUAL CLASSIFIER DEMO")
    print("="*70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = create_demo_data()
    
    # Demo 1: CSP Classifier
    print("\\n1. CSP CLASSIFIER EXAMPLE")
    print("-" * 30)
    
    try:
        csp = CSPClassifier(n_components=4, clf_type='lda')
        csp_results = csp.train(X_train, y_train, X_val, y_val, X_test, y_test, verbose=True)
        
        # Make some predictions
        new_predictions = csp.predict(X_test[:5])
        print(f"Sample predictions on 5 test samples: {new_predictions}")
        
    except Exception as e:
        print(f"CSP demo failed: {e}")
    
    # Demo 2: EEGNet Classifier  
    print("\\n\\n2. EEGNET CLASSIFIER EXAMPLE")
    print("-" * 30)
    
    try:
        eegnet = EEGNetClassifier(dropoutRate=0.3)
        eegnet_results = eegnet.train(X_train, y_train, X_val, y_val, X_test, y_test,
                                    lr=1e-3, epochs=5, batch_size=16, verbose=True)
        
        # Get probabilities
        probs = eegnet.predict_proba(X_test[:3])
        print(f"Sample prediction probabilities: {probs}")
        
    except Exception as e:
        print(f"EEGNet demo failed: {e}")
    
    # Demo 3: CBraMod Classifier (will require pretrained weights)
    print("\\n\\n3. CBRAMOD CLASSIFIER EXAMPLE")
    print("-" * 30)
    
    try:
        # Need to reshape data for CBraMod (patches format)
        # This is a simplified version - normally use prepare_data_for_cbramod
        n_patches = X_train.shape[2] // 200
        if n_patches > 0:
            X_train_patches = X_train[:, :, :n_patches*200].reshape(
                X_train.shape[0], X_train.shape[1], n_patches, 200)
            X_val_patches = X_val[:, :, :n_patches*200].reshape(
                X_val.shape[0], X_val.shape[1], n_patches, 200)
            X_test_patches = X_test[:, :, :n_patches*200].reshape(
                X_test.shape[0], X_test.shape[1], n_patches, 200)
            
            cbramod = CBraModClassifier(dropout=0.1)
            cbramod_results = cbramod.train(X_train_patches, y_train, X_val_patches, y_val, 
                                          X_test_patches, y_test, lr=1e-4, epochs=3, 
                                          batch_size=8, verbose=True)
        else:
            print("Data too short for CBraMod patches (need at least 200 time points)")
            
    except Exception as e:
        print(f"CBraMod demo failed: {e}")

def demo_comparison():
    """Demonstrate comparing multiple classifiers"""
    print("\\n\\n" + "="*70)
    print("CLASSIFIER COMPARISON DEMO")
    print("="*70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = create_demo_data()
    
    # Configure classifiers for quick demo
    configs = {
        'csp': {'n_components': 3, 'clf_type': 'lda'},
        'eegnet': {'dropoutRate': 0.4, 'lr': 1e-3, 'epochs': 3, 'batch_size': 16}
        # Skip CBraMod for demo due to pretrained weights requirement
    }
    
    try:
        results = compare_classifiers(X_train, y_train, X_val, y_val, X_test, y_test,
                                    classifiers_config=configs, verbose=True)
        
        print("\\nComparison completed! Results stored in 'results' dictionary.")
        
    except Exception as e:
        print(f"Comparison demo failed: {e}")

if __name__ == "__main__":
    print("EEG CLASSIFIER ARCHITECTURE DEMO")
    print("="*70)
    print("This demo shows the new unified classifier interface.")
    print("All classifiers now have consistent methods: train(), predict(), evaluate(), etc.")
    print()
    
    # Run individual demos
    demo_individual_classifiers()
    
    # Run comparison demo
    demo_comparison()
    
    print("\\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\\nThe new architecture provides:")
    print("✅ Consistent interface across all models")
    print("✅ Standardized evaluation metrics")
    print("✅ Easy model comparison")
    print("✅ Proper class-based organization")
    print("✅ Error handling and validation")
    print("\\nYou can now use any classifier with the same methods!")