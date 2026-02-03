"""
CSP Classifier with integrated training and data preparation
"""

import numpy as np
import mne
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from classification.base_classifier import BaseEEGClassifier
from classification.models.csp.csp import CSPClassifier as CSPModel
from data.data_configs import EEG_SAMPLE_FREQUENCY

class CSPClassifier(BaseEEGClassifier):
    """CSP classifier with integrated training and data preparation"""
    
    def __init__(self, n_components=4, clf_type='lda', **csp_params):
        super().__init__(name="CSP")
        self.n_components = n_components
        self.clf_type = clf_type
        self.csp_params = csp_params
        
    def prepare_data(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        """Bandpass filter EEG data for CSP"""
        def filter_data(X, y):
            if len(X) == 0:
                return X, y
            
            n_trials, n_chans, n_times = X.shape
            info = mne.create_info(ch_names=n_chans, sfreq=EEG_SAMPLE_FREQUENCY, ch_types=['eeg']*n_chans)
            epochs = mne.EpochsArray(X, info, verbose=False)
            epochs.filter(l_freq=0.5, h_freq=40.0, verbose=False)
            X_clean = epochs.get_data()
            
            if X_clean.ndim == 4:
                X_clean = X_clean.squeeze(-1)
                
            return X_clean, np.array(y)
        
        X_train_clean, y_train_clean = filter_data(X_train, y_train)
        X_val_clean, y_val_clean = filter_data(X_val, y_val)
        
        if X_test is not None:
            X_test_clean, y_test_clean = filter_data(X_test, y_test)
        else:
            X_test_clean, y_test_clean = None, None
            
        return X_train_clean, y_train_clean, X_val_clean, y_val_clean, X_test_clean, y_test_clean
    
    def train(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, verbose=True):
        """Train CSP classifier"""
        if verbose:
            print(f"\\n[CSP Training] Components: {self.n_components}, Classifier: {self.clf_type.upper()}")
        
        # Prepare data
        X_train_prep, y_train_prep, X_val_prep, y_val_prep, X_test_prep, y_test_prep = self.prepare_data(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Build and train model
        self.model = CSPModel(n_components=self.n_components, clf_type=self.clf_type, **self.csp_params)
        self.model.fit(X_train_prep, y_train_prep)
        self.is_trained = True
        
        # Evaluate
        if X_test_prep is not None:
            results = self.evaluate(X_test_prep, y_test_prep, "Test")
        else:
            results = self.evaluate(X_val_prep, y_val_prep, "Validation")
        
        return results
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            preds = self.predict(X)
            probs = np.zeros((len(preds), 2))
            probs[preds == 0, 0] = 1.0
            probs[preds == 1, 1] = 1.0
            return probs
    
    def optimize(self, X_train, y_train, X_val, y_val, param_grid=None, verbose=True):
        """Optimize CSP parameters"""
        if param_grid is None:
            # Separate optimization for each classifier type
            lda_grid = [
                {'n_components': n_comp, 'clf_type': 'lda', 'reg': reg}
                for n_comp in [4, 6, 8]  # CSP components
                for reg in [None, 'auto', 0.1]  # CSP regularization
            ]
            
            rf_grid = [
                {'n_components': n_comp, 'clf_type': 'rf', 'max_depth': depth, 'n_estimators': n_est}
                for n_comp in [4, 6, 8]  # CSP components  
                for depth in [3, 5, 7]  # RF max depth
                for n_est in [50, 100]  # RF estimators
            ]
            
            param_grid = lda_grid + rf_grid  # Total: 9 + 18 = 27 combinations
        
        best_score = 0
        best_params = None
        
        if verbose:
            print(f"\\n[CSP Optimization] Testing {len(param_grid)} combinations...")
        
        for i, params in enumerate(param_grid):
            try:
                classifier = CSPClassifier(**params)
                results = classifier.train(X_train, y_train, X_val, y_val, verbose=False)
                score = results['balanced_acc']
                
                if verbose:
                    print(f"  {i+1:2d}. {params} → {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                if verbose:
                    print(f"  {i+1:2d}. {params} → Failed: {str(e)[:50]}")
        
        if verbose:
            print(f"\\nBest: {best_params} (Score: {best_score:.4f})")
        
        return {'best_params': best_params, 'best_score': best_score}