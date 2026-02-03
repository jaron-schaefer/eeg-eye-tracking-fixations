"""
Unified EEG Classification Architecture

Base class and interface for all EEG classification models with integrated
data preparation, training, evaluation, and parameter optimization.
"""

import abc
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import (
    balanced_accuracy_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)

class BaseEEGClassifier(abc.ABC):
    """Abstract base class for all EEG classifiers with unified interface"""
    
    def __init__(self, name: str):
        """Initialize base classifier"""
        self.name = name
        self.model = None
        self.is_trained = False
        self.training_history = {}
    
    @abc.abstractmethod
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: Optional[np.ndarray] = None, 
                    y_test: Optional[np.ndarray] = None) -> Tuple[Any, ...]:
        """Preprocess data for model-specific format"""
        pass
    
    @abc.abstractmethod
    def build_model(self, input_shape: Tuple, **kwargs) -> Any:
        """Build the model architecture"""
        pass
    
    @abc.abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray, 
             X_test: Optional[np.ndarray] = None,
             y_test: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """Train the model with validation monitoring"""
        pass
    
    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions (0 or 1)"""
        pass
    
    @abc.abstractmethod
    def optimize_parameters(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Find optimal hyperparameters using train/val data"""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, set_name: str = "Evaluation") -> Dict[str, Any]:
        """Standardized evaluation with all metrics"""
        if not self.is_trained:
            raise ValueError(f"{self.name} model has not been trained yet")
        
        preds = self.predict(X)
        probs = self.predict_proba(X) if hasattr(self, 'predict_proba') else None
        
        # Calculate metrics
        bal_acc = balanced_accuracy_score(y, preds)
        conf_matrix = confusion_matrix(y, preds)
        report = classification_report(y, preds, target_names=['Non-Target', 'Target'], zero_division=0)
        
        # AUROC if probabilities available
        auroc = None
        if probs is not None:
            try:
                if probs.ndim == 2:
                    probs_positive = probs[:, 1]
                else:
                    probs_positive = probs
                auroc = roc_auc_score(y, probs_positive)
            except ValueError:
                auroc = 0.5
        
        results = {
            'score': bal_acc,
            'balanced_acc': bal_acc,
            'auroc': auroc,
            'confusion_matrix': conf_matrix,
            'report': report,
            'predictions': preds,
            'probabilities': probs
        }
        
        # Print results
        print(f"\\n{'='*60}")
        print(f"{self.name.upper()} - {set_name.upper()} RESULTS")
        print(f"{'='*60}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        if auroc is not None:
            print(f"AUROC Score:      {auroc:.4f}")
        
        print(f"\\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Non-Target  Target")
        print(f"Actual Non-Target    {conf_matrix[0,0]:4d}     {conf_matrix[0,1]:4d}")
        print(f"Actual Target        {conf_matrix[1,0]:4d}     {conf_matrix[1,1]:4d}")
        
        # Check predictions
        unique_preds = np.unique(preds)
        if len(unique_preds) == 1:
            print(f"\\n⚠️  WARNING: Only predicted class {unique_preds[0]}")
        else:
            pred_counts = dict(zip(*np.unique(preds, return_counts=True)))
            print(f"\\n✅ Predictions: {pred_counts}")
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get classifier information and state"""
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'model_type': type(self.model).__name__ if self.model else None,
            'training_history': self.training_history
        }