"""
Feature-based EEG Classifier with Model Comparison

This classifier uses extracted EEG features for classification and includes
functionality to compare different machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix, 
    f1_score, accuracy_score, make_scorer
)
import warnings
warnings.filterwarnings('ignore')

from classification.base_classifier import BaseEEGClassifier
from data.data_loader import DataLoader
from data.data_configs import SUBJECTS

class FeatureClassifier(BaseEEGClassifier):
    """Feature-based EEG classifier with model comparison capabilities"""
    
    AVAILABLE_MODELS = {
        'rf': RandomForestClassifier,
        'svm': SVC,
    }
    
    DEFAULT_PARAMS = {
        'rf': {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'},
        'svm': {'kernel': 'rbf', 'random_state': 42, 'class_weight': 'balanced', 'probability': True},
    }
    
    def __init__(self, model_type: str = 'rf', feature_mode: str = 'advanced', 
                 subjects: List[str] = None, **model_params):
        """Initialize Feature Classifier with specified model type and parameters"""
        super().__init__(name=f"Feature-{model_type.upper()}")
        
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.feature_mode = feature_mode
        self.subjects = subjects or SUBJECTS
        
        # Merge default params with user params
        params = self.DEFAULT_PARAMS[model_type].copy()
        params.update(model_params)
        self.model_params = params
        
        # Data preprocessing components
        self.imputer = SimpleImputer(strategy='mean')
        self.scalers = {}  # Per-subject scalers
        self.feature_names = None
        self.feature_names_clean = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load EEG features data"""
        return DataLoader.load_features(self.subjects, mode=self.feature_mode)
    
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_eval: np.ndarray, y_eval: np.ndarray) -> Tuple[Any, ...]:
        """Prepare features data with cleaning, imputation, and scaling"""
        
        # For features, we combine train and eval data for consistent preprocessing
        X_all = np.vstack([X_train, X_eval])
        y_all = np.hstack([y_train, y_eval])
        
        # Clean data - handle NaNs
        nan_counts = np.isnan(X_all).sum(axis=0)
        total_samples = X_all.shape[0]
        
        # Remove features with >50% NaN
        keep_features = nan_counts <= (total_samples * 0.5)
        X_clean = X_all[:, keep_features]
        
        # Impute remaining NaNs
        if not hasattr(self.imputer, 'statistics_'):
            X_imputed = self.imputer.fit_transform(X_clean)
        else:
            X_imputed = self.imputer.transform(X_clean)
        
        # Split back into original sets
        train_size = len(X_train)
        
        X_train_clean = X_imputed[:train_size]
        X_eval_clean = X_imputed[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_eval_scaled = scaler.transform(X_eval_clean)
        
        return X_train_scaled, y_train, X_eval_scaled, y_eval
    
    def build_model(self, input_shape: Tuple, **kwargs) -> Any:
        """Build the specified model"""
        model_class = self.AVAILABLE_MODELS[self.model_type]
        return model_class(**self.model_params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_eval: np.ndarray, y_eval: np.ndarray, 
             **kwargs) -> Dict[str, Any]:
        """Train the feature classifier"""
        
        print(f"\n[{self.name} Training] Model: {self.model_type.upper()}")
        print(f"Training samples: {len(X_train)}, Evaluation samples: {len(X_eval)}")
        
        # Prepare data (includes cleaning, imputation, and scaling)
        X_train_prep, y_train_prep, X_eval_prep, y_eval_prep = self.prepare_data(
            X_train, y_train, X_eval, y_eval
        )
        
        # Build and train model
        self.model = self.build_model(input_shape=X_train_prep.shape)
        self.model.fit(X_train_prep, y_train_prep)
        self.is_trained = True
        
        # Evaluate using base class method
        results = self.evaluate(X_eval_prep, y_eval_prep)
        
        return results
    
    def _scale_per_subject(self, X_train, X_eval, 
                                    subject_ids_train, subject_ids_eval):
        """Scale features per subject (simplified version)"""
        X_train_scaled = np.zeros_like(X_train)
        X_eval_scaled = np.zeros_like(X_eval)
        
        # Train scalers on training subjects
        subject_ids_train = np.array(subject_ids_train)
        for subj in np.unique(subject_ids_train):
            subj_idx = np.where(subject_ids_train == subj)[0]
            scaler = StandardScaler()
            X_train_scaled[subj_idx] = scaler.fit_transform(X_train[subj_idx])
            self.scalers[subj] = scaler
        
        # Apply to evaluation set
        if subject_ids_eval is not None:
            subject_ids_eval = np.array(subject_ids_eval)
            for subj in np.unique(subject_ids_eval):
                subj_idx = np.where(subject_ids_eval == subj)[0]
                if subj in self.scalers:
                    X_eval_scaled[subj_idx] = self.scalers[subj].transform(X_eval[subj_idx])
                else:
                    # Use global scaler if subject not in training
                    scaler = StandardScaler()
                    X_eval_scaled[subj_idx] = scaler.fit_transform(X_eval[subj_idx])
        
        return X_train_scaled, X_eval_scaled

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        return self.model.predict(X)