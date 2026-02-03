"""
Feature-based classifiers for EEG and Eye Tracking data
"""

from .feature_classifier import FeatureClassifier
from .et_feature_classifier import ETFeatureClassifier

__all__ = ['FeatureClassifier', 'ETFeatureClassifier']