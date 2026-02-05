"""
Eye Tracking Feature-based Classifier with Model Comparison

This classifier uses extracted eye tracking features for classification and includes
functionality to compare different machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix, 
    f1_score, accuracy_score, make_scorer
)
import warnings
warnings.filterwarnings('ignore')

from classification.base_classifier import BaseEEGClassifier
from data.data_loader import DataLoader
from data.data_configs import SUBJECTS

class ETFeatureClassifier(BaseEEGClassifier):
    """Eye Tracking feature-based classifier with model comparison capabilities"""
    
    AVAILABLE_MODELS = {
        'rf': RandomForestClassifier,
        'svm': SVC
    }
    
    DEFAULT_PARAMS = {
        'rf': {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'},
        'svm': {'kernel': 'rbf', 'random_state': 42, 'class_weight': 'balanced', 'probability': True}
    }
    
    def __init__(self, model_type: str = 'rf', subjects: List[str] = None, **model_params):
        """Initialize ET Feature Classifier with specified model type and parameters"""
        super().__init__(name=f"ET-Feature-{model_type.upper()}")
        
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.subjects = subjects or SUBJECTS
        
        # Merge default params with user params
        params = self.DEFAULT_PARAMS[model_type].copy()
        params.update(model_params)
        self.model_params = params
        
        # Data preprocessing components
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_names_clean = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load ET features data"""
        return DataLoader.load_et_features(self.subjects)
    
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_eval: np.ndarray, y_eval: np.ndarray) -> Tuple[Any, ...]:
        """Prepare ET features data with cleaning, imputation, and scaling"""
        
        # For features, we don't have separate train/eval splits initially
        # Combine all data for preprocessing
        X_all = np.vstack([X_train, X_eval])
        y_all = np.hstack([y_train, y_eval])
        
        # Clean data - handle NaNs
        nan_counts = np.isnan(X_all).sum(axis=0)
        total_samples = X_all.shape[0]
        
        print(f"Features with NaNs: {(nan_counts > 0).sum()}/{X_all.shape[1]}")
        
        # Remove features with >50% NaN
        keep_features = nan_counts <= (total_samples * 0.5)
        X_clean = X_all[:, keep_features]
        
        print(f"Features after cleaning: {X_clean.shape[1]}/{X_all.shape[1]}")
        
        # Impute remaining NaNs
        if not hasattr(self.imputer, 'statistics_'):
            X_imputed = self.imputer.fit_transform(X_clean)
        else:
            X_imputed = self.imputer.transform(X_clean)
        
        # Split back into original sets
        train_size = len(X_train)
        eval_size = len(X_eval)
        
        X_train_clean = X_imputed[:train_size]
        X_eval_clean = X_imputed[train_size:train_size+eval_size]
        
        return X_train_clean, y_train, X_eval_clean, y_eval
    
    def build_model(self, input_shape: Tuple, **kwargs) -> Any:
        """Build the specified model"""
        model_class = self.AVAILABLE_MODELS[self.model_type]
        return model_class(**self.model_params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_eval: np.ndarray, y_eval: np.ndarray,
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
             **kwargs) -> Dict[str, Any]:
        """Train the ET feature classifier"""
        
        print(f"\n[{self.name} Training] Model: {self.model_type.upper()}")
        print(f"Training samples: {len(X_train)}, Evaluation samples: {len(X_eval)}")
        
        # Prepare data
        X_train_prep, y_train_prep, X_eval_prep, y_eval_prep = self.prepare_data(
            X_train, y_train, X_eval, y_eval
        )
        
        X_train_prep = self.scaler.fit_transform(X_train_prep)
            
        X_eval_prep = self.scaler.transform(X_eval_prep)
        
        # Build and train model
        self.model = self.build_model(input_shape=X_train_prep.shape)
        self.model.fit(X_train_prep, y_train_prep)
        self.is_trained = True
        
        # Evaluate using base class method (it calls predict internally)
        results = self.evaluate(X_eval_prep, y_eval_prep)
        
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
