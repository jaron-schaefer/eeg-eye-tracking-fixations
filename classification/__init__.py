"""
EEG Classification Models

This package contains various classification models for EEG data analysis including:
- Deep learning models (EEGNet, CBRAMod)
- Traditional signal processing approaches (CSP)
- Feature-based classifiers (Random Forest, SVM, etc.)
"""

# Import base classifier
from .base_classifier import BaseEEGClassifier

# Import specific model implementations
from .models.csp.csp_classifier import CSPClassifier
from .models.eeg_net.eegnet_classifier import EEGNetClassifier 
from .models.cbramod.cbramod_classifier import CBraModClassifier

# Import feature-based classifiers
from .models.features.feature_classifier import FeatureClassifier
from .models.features.et_feature_classifier import ETFeatureClassifier

__all__ = [
    'BaseEEGClassifier',
    'CSPClassifier', 
    'EEGNetClassifier',
    'CBraModClassifier',
    'FeatureClassifier',
    'ETFeatureClassifier'
]