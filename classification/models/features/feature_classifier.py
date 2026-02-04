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
import json
import json
import ast

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
             X_eval: np.ndarray, y_eval: np.ndarray, eval_stimulus_summaries: Optional[List[str]] = None,
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
        
        if eval_stimulus_summaries is not None:
            # Evaluate with stimulus summaries
            results = self.evaluate_with_summaries(
                X_eval_prep, y_eval_prep, eval_stimulus_summaries
            )
        else:
            # Evaluate using base class method
            results = self.evaluate(X_eval_prep, y_eval_prep)
        
        return results
    
    def evaluate_with_summaries(self, X_eval, y_eval, eval_stimulus_summaries):
        """Evaluate model using stimulus summaries"""
        if not self.is_trained:
            raise ValueError(f"{self.name} model has not been trained yet")
        
        # Make predictions
        preds = self.predict(X_eval)
        
        # Simple parsing: list of JSON strings -> dicts -> DataFrame -> use 'n_airplanes' as summary key
        parsed_items = []
        for it in eval_stimulus_summaries:
            if isinstance(it, str):
                try:
                    parsed_items.append(json.loads(it))
                except Exception:
                    # Fallback: keep as plain string if not valid JSON
                    parsed_items.append({'raw': it})
            elif isinstance(it, dict):
                parsed_items.append(it)
            else:
                parsed_items.append({'raw': str(it)})

        df_meta = pd.DataFrame(parsed_items)
        print(df_meta.head())
        if 'n_airplanes' in df_meta.columns:
            summaries_extracted = df_meta['n_airplanes'].tolist()
        else:
            # Fallback: use raw string
            summaries_extracted = df_meta.get('raw', pd.Series([str(x) for x in eval_stimulus_summaries])).tolist()

        # Aggregate predictions by stimulus summary
        df_results = pd.DataFrame({
            'stimulus_summary': summaries_extracted,
            'true_label': y_eval,
            'predicted_label': preds
        })
        
        # Compute per-group accuracies (fraction of correct predictions within each group)
        grouped = df_results.groupby('stimulus_summary')
        per_group_acc = grouped.apply(lambda df: float((df['predicted_label'] == df['true_label']).mean()))
        
        # Compute per-group label counts (true vs predicted) and group sizes
        per_group_counts = {}
        for grp, gdf in grouped:
            true_counts = gdf['true_label'].value_counts().to_dict()
            pred_counts = gdf['predicted_label'].value_counts().to_dict()
            per_group_counts[grp] = {
                'n': int(len(gdf)),
                'true': true_counts,
                'pred': pred_counts,
            }
        
        summary_preds = df_results.groupby('stimulus_summary')['predicted_label'].agg(lambda x: np.bincount(x).argmax())
        summary_true = df_results.groupby('stimulus_summary')['true_label'].first()
        
        # Calculate metrics
        bal_acc = balanced_accuracy_score(summary_true, summary_preds)
        conf_matrix = confusion_matrix(summary_true, summary_preds)
        report = classification_report(summary_true, summary_preds, target_names=['Non-Target', 'Target'], zero_division=0)
        
        results = {
            'score': bal_acc,
            'balanced_acc': bal_acc,
            'confusion_matrix': conf_matrix,
            'report': report,
            'predictions': summary_preds.values,
            'per_group_accuracy': per_group_acc.to_dict(),
            'per_group_counts': per_group_counts
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"{self.name.upper()} - EVALUATION RESULTS (Stimulus Summary)")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print("Per-group Accuracies and Label Counts:")
        for grp, acc in per_group_acc.items():
            counts = per_group_counts.get(grp, {'n': 0, 'true': {}, 'pred': {}})
            true0 = counts['true'].get(0, counts['true'].get('0', 0))
            true1 = counts['true'].get(1, counts['true'].get('1', 0))
            pred0 = counts['pred'].get(0, counts['pred'].get('0', 0))
            pred1 = counts['pred'].get(1, counts['pred'].get('1', 0))
            print(f"  Group {grp}: acc={acc:.3f}, n={counts['n']}, true[0]={true0}, true[1]={true1}, pred[0]={pred0}, pred[1]={pred1}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(report)
        
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