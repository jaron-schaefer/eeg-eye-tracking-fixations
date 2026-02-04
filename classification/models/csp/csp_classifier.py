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
        
    def prepare_data(self, X_train, y_train, X_eval, y_eval):
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
        X_eval_clean, y_eval_clean = filter_data(X_eval, y_eval)
            
        return X_train_clean, y_train_clean, X_eval_clean, y_eval_clean
    
    def train(self, X_train, y_train, X_eval, y_eval, verbose=True):
        """Train CSP classifier"""
        if verbose:
            print(f"\\n[CSP Training] Components: {self.n_components}, Classifier: {self.clf_type.upper()}")
        
        # Prepare data
        X_train_prep, y_train_prep, X_eval_prep, y_eval_prep = self.prepare_data(
            X_train, y_train, X_eval, y_eval
        )
        
        # Build and train model
        self.model = CSPModel(n_components=self.n_components, clf_type=self.clf_type, **self.csp_params)
        self.model.fit(X_train_prep, y_train_prep)
        self.is_trained = True
        
        # Evaluate on evaluation data
        results = self.evaluate(X_eval_prep, y_eval_prep, "Evaluation")
        
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