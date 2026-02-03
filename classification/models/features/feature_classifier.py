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

class FeatureClassifier(BaseEEGClassifier):
    """Feature-based EEG classifier with model comparison capabilities"""
    
    AVAILABLE_MODELS = {
        'rf': RandomForestClassifier,
        'svm': SVC,
        'lr': LogisticRegression,
        'knn': KNeighborsClassifier,
        'nb': GaussianNB,
        'dt': DecisionTreeClassifier
    }
    
    DEFAULT_PARAMS = {
        'rf': {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'},
        'svm': {'kernel': 'rbf', 'random_state': 42, 'class_weight': 'balanced', 'probability': True},
        'lr': {'random_state': 42, 'class_weight': 'balanced', 'max_iter': 1000},
        'knn': {'n_neighbors': 5},
        'nb': {},
        'dt': {'random_state': 42, 'class_weight': 'balanced'}
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
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: Optional[np.ndarray] = None, 
                    y_test: Optional[np.ndarray] = None) -> Tuple[Any, ...]:
        """Prepare features data with cleaning, imputation, and scaling"""
        
        # For features, we don't have separate train/val splits initially
        # Combine all data for preprocessing
        X_all = np.vstack([X_train, X_val] + ([X_test] if X_test is not None else []))
        y_all = np.hstack([y_train, y_val] + ([y_test] if y_test is not None else []))
        
        # Get feature names if available
        if self.feature_names is None:
            # Try to get feature names from data loader
            try:
                _, _, self.feature_names, _ = self.load_data()
            except:
                self.feature_names = [f'feature_{i}' for i in range(X_all.shape[1])]
        
        # Clean data - handle NaNs
        df = pd.DataFrame(X_all, columns=self.feature_names)
        nan_counts = df.isna().sum()
        
        # Remove features with >50% NaN
        high_nan = nan_counts[nan_counts > len(df) * 0.5].index.tolist()
        self.feature_names_clean = [f for f in self.feature_names if f not in high_nan]
        
        # Get clean features
        X_clean = df[self.feature_names_clean].values
        
        # Impute remaining NaNs
        if not hasattr(self.imputer, 'statistics_'):
            X_imputed = self.imputer.fit_transform(X_clean)
        else:
            X_imputed = self.imputer.transform(X_clean)
        
        # Split back into original sets
        train_size = len(X_train)
        val_size = len(X_val)
        
        X_train_clean = X_imputed[:train_size]
        X_val_clean = X_imputed[train_size:train_size+val_size]
        
        if X_test is not None:
            X_test_clean = X_imputed[train_size+val_size:]
        else:
            X_test_clean = None
        
        return X_train_clean, y_train, X_val_clean, y_val, X_test_clean, y_test
    
    def build_model(self, input_shape: Tuple, **kwargs) -> Any:
        """Build the specified model"""
        model_class = self.AVAILABLE_MODELS[self.model_type]
        return model_class(**self.model_params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray, 
             X_test: Optional[np.ndarray] = None,
             y_test: Optional[np.ndarray] = None,
             subject_ids_train: Optional[List[str]] = None,
             subject_ids_val: Optional[List[str]] = None,
             subject_ids_test: Optional[List[str]] = None,
             **kwargs) -> Dict[str, Any]:
        """Train the feature classifier"""
        
        print(f"\\n[{self.name} Training] Model: {self.model_type.upper()}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Prepare data
        X_train_prep, y_train_prep, X_val_prep, y_val_prep, X_test_prep, y_test_prep = self.prepare_data(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Scale features per subject if subject IDs provided
        if subject_ids_train is not None:
            X_train_prep, X_val_prep, X_test_prep = self._scale_per_subject(
                X_train_prep, X_val_prep, X_test_prep,
                subject_ids_train, subject_ids_val, subject_ids_test
            )
        else:
            # Global scaling
            scaler = StandardScaler()
            X_train_prep = scaler.fit_transform(X_train_prep)
            X_val_prep = scaler.transform(X_val_prep)
            if X_test_prep is not None:
                X_test_prep = scaler.transform(X_test_prep)
        
        # Build and train model
        self.model = self.build_model(input_shape=X_train_prep.shape)
        self.model.fit(X_train_prep, y_train_prep)
        self.is_trained = True
        
        # Evaluate
        results = {}
        results['train'] = self._evaluate_set(X_train_prep, y_train_prep, "Train")
        results['validation'] = self._evaluate_set(X_val_prep, y_val_prep, "Validation")
        
        if X_test_prep is not None:
            results['test'] = self._evaluate_set(X_test_prep, y_test_prep, "Test")
        
        return results
    
    def _scale_per_subject(self, X_train, X_val, X_test, 
                          subject_ids_train, subject_ids_val, subject_ids_test):
        """Scale features per subject"""
        X_train_scaled = np.zeros_like(X_train)
        X_val_scaled = np.zeros_like(X_val)
        X_test_scaled = np.zeros_like(X_test) if X_test is not None else None
        
        # Train scalers on training subjects
        subject_ids_train = np.array(subject_ids_train)
        for subj in np.unique(subject_ids_train):
            subj_idx = np.where(subject_ids_train == subj)[0]
            scaler = StandardScaler()
            X_train_scaled[subj_idx] = scaler.fit_transform(X_train[subj_idx])
            self.scalers[subj] = scaler
        
        # Apply to validation set
        if subject_ids_val is not None:
            subject_ids_val = np.array(subject_ids_val)
            for subj in np.unique(subject_ids_val):
                subj_idx = np.where(subject_ids_val == subj)[0]
                if subj in self.scalers:
                    X_val_scaled[subj_idx] = self.scalers[subj].transform(X_val[subj_idx])
                else:
                    # Use global scaler if subject not in training
                    scaler = StandardScaler()
                    X_val_scaled[subj_idx] = scaler.fit_transform(X_val[subj_idx])
        
        # Apply to test set
        if X_test is not None and subject_ids_test is not None:
            subject_ids_test = np.array(subject_ids_test)
            for subj in np.unique(subject_ids_test):
                subj_idx = np.where(subject_ids_test == subj)[0]
                if subj in self.scalers:
                    X_test_scaled[subj_idx] = self.scalers[subj].transform(X_test[subj_idx])
                else:
                    scaler = StandardScaler()
                    X_test_scaled[subj_idx] = scaler.fit_transform(X_test[subj_idx])
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def _evaluate_set(self, X, y, set_name: str) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        y_pred = self.model.predict(X)
        
        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        bal_acc = balanced_accuracy_score(y, y_pred)
        
        print(f"{set_name}: Accuracy={acc:.4f}, Macro-F1={f1_macro:.4f}, Balanced Acc={bal_acc:.4f}")
        
        return {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'balanced_accuracy': bal_acc,
            'predictions': y_pred
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Apply same preprocessing as training
        if hasattr(self.imputer, 'statistics_'):
            df = pd.DataFrame(X, columns=self.feature_names)
            X_clean = df[self.feature_names_clean].values
            X_imputed = self.imputer.transform(X_clean)
            return self.model.predict(X_imputed)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'predict_proba'):
            if hasattr(self.imputer, 'statistics_'):
                df = pd.DataFrame(X, columns=self.feature_names)
                X_clean = df[self.feature_names_clean].values
                X_imputed = self.imputer.transform(X_clean)
                return self.model.predict_proba(X_imputed)
            else:
                return self.model.predict_proba(X)
        else:
            raise ValueError(f"Model {self.model_type} does not support probability prediction")
    
    def cross_validate(self, cv_type: str = 'stratified', n_splits: int = 5, 
                      subjects: List[str] = None, 
                      subject_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform cross-validation with stratified or grouped splits"""
        if subjects is None:
            subjects = self.subjects
        
        # Load data
        X, y, feature_names, subject_ids_loaded = DataLoader.load_features(subjects, mode=self.feature_mode)
        self.feature_names = feature_names
        
        if subject_ids is None:
            subject_ids = subject_ids_loaded
        
        # Prepare data (preprocessing)
        X_prep, _, _, _, _, _ = self.prepare_data(X, y, np.array([]), np.array([]))
        
        # Per-subject scaling
        if cv_type == 'grouped' and subject_ids is not None:
            subject_ids = np.array(subject_ids)
            X_scaled = np.zeros_like(X_prep)
            for subj in np.unique(subject_ids):
                subj_idx = np.where(subject_ids == subj)[0]
                scaler = StandardScaler()
                X_scaled[subj_idx] = scaler.fit_transform(X_prep[subj_idx])
            X_prep = X_scaled
        else:
            # Global scaling
            scaler = StandardScaler()
            X_prep = scaler.fit_transform(X_prep)
        
        # Set up cross-validation
        if cv_type == 'stratified':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_splits = cv.split(X_prep, y)
        elif cv_type == 'grouped':
            if subject_ids is None:
                raise ValueError("subject_ids required for grouped CV")
            cv = GroupKFold(n_splits=n_splits)
            cv_splits = cv.split(X_prep, y, groups=subject_ids)
        else:
            raise ValueError(f"Unknown cv_type: {cv_type}")
        
        # Perform CV
        print(f"\\n=== {cv_type.title()} K-Fold CV ({n_splits} splits) ===")
        print(f"Model: {self.model_type.upper()}")
        
        y_true_all, y_pred_all = [], []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
            # Build and train model
            model = self.build_model(input_shape=X_prep.shape)
            model.fit(X_prep[train_idx], y[train_idx])
            
            # Predict
            y_pred = model.predict(X_prep[test_idx])
            
            # Store for overall metrics
            y_true_all.append(y[test_idx])
            y_pred_all.append(y_pred)
            
            # Compute fold metrics
            acc = accuracy_score(y[test_idx], y_pred)
            f1 = f1_score(y[test_idx], y_pred, average='macro')
            bal_acc = balanced_accuracy_score(y[test_idx], y_pred)
            
            fold_result = {
                'fold': fold,
                'accuracy': acc,
                'f1_macro': f1,
                'balanced_accuracy': bal_acc
            }
            fold_results.append(fold_result)
            
            if cv_type == 'grouped' and subject_ids is not None:
                subjects_in_fold = np.unique(subject_ids[test_idx])
                print(f"Fold {fold} | Subjects: {subjects_in_fold} | Accuracy={acc:.4f}, Macro-F1={f1:.4f}, Balanced Acc={bal_acc:.4f}")
            else:
                print(f"Fold {fold}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}, Balanced Acc={bal_acc:.4f}")
        
        # Aggregate results
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        
        overall_f1 = f1_score(y_true_all, y_pred_all, average='macro')
        overall_bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        overall_acc = accuracy_score(y_true_all, y_pred_all)
        
        print(f"\\nOverall Results:")
        print(f"Accuracy: {overall_acc:.4f}")
        print(f"Macro-F1: {overall_f1:.4f}")
        print(f"Balanced Accuracy: {overall_bal_acc:.4f}")
        
        return {
            'cv_type': cv_type,
            'n_splits': n_splits,
            'model_type': self.model_type,
            'fold_results': fold_results,
            'overall_accuracy': overall_acc,
            'overall_f1_macro': overall_f1,
            'overall_balanced_accuracy': overall_bal_acc,
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'confusion_matrix': confusion_matrix(y_true_all, y_pred_all)
        }
    
    @classmethod
    def compare_models(cls, model_types: List[str] = None, cv_type: str = 'stratified',
                      n_splits: int = 5, feature_mode: str = 'advanced',
                      subjects: List[str] = None) -> pd.DataFrame:
        """Compare multiple models using cross-validation"""
        if model_types is None:
            model_types = ['rf', 'svm', 'lr']
        
        if subjects is None:
            subjects = SUBJECTS
        
        print(f"\\n{'='*60}")
        print(f"MODEL COMPARISON: {model_types}")
        print(f"CV Type: {cv_type}, Splits: {n_splits}, Feature Mode: {feature_mode}")
        print(f"Subjects: {subjects}")
        print(f"{'='*60}")
        
        results = []
        
        for model_type in model_types:
            print(f"\\n{'-'*40}")
            print(f"Testing {model_type.upper()}")
            print(f"{'-'*40}")
            
            classifier = cls(model_type=model_type, feature_mode=feature_mode, subjects=subjects)
            cv_results = classifier.cross_validate(cv_type=cv_type, n_splits=n_splits)
            
            results.append({
                'Model': model_type.upper(),
                'Accuracy': cv_results['overall_accuracy'],
                'F1-Macro': cv_results['overall_f1_macro'],
                'Balanced Accuracy': cv_results['overall_balanced_accuracy'],
                'Accuracy Std': np.std([r['accuracy'] for r in cv_results['fold_results']]),
                'F1-Macro Std': np.std([r['f1_macro'] for r in cv_results['fold_results']]),
                'Balanced Acc Std': np.std([r['balanced_accuracy'] for r in cv_results['fold_results']])
            })
        
        # Create comparison DataFrame
        df_results = pd.DataFrame(results)
        
        # Sort by balanced accuracy
        df_results = df_results.sort_values('Balanced Accuracy', ascending=False)
        
        print(f"\\n{'='*60}")
        print("FINAL COMPARISON RESULTS")
        print(f"{'='*60}")
        print(df_results.round(4).to_string(index=False))
        
        # Find best model
        best_model = df_results.iloc[0]['Model'].lower()
        best_score = df_results.iloc[0]['Balanced Accuracy']
        print(f"\\nBest Model: {best_model.upper()} (Balanced Accuracy: {best_score:.4f})")
        
        return df_results