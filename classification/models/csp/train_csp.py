import numpy as np
from sklearn.metrics import (balanced_accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score)
from classification.models.csp.csp import CSPClassifier

def train_csp(X_train, y_train, 
              X_val, y_val,             
              X_test=None, y_test=None, 
              n_components=4,           
              clf_type='lda'):          
    
    print(f"Training CSP-{clf_type.upper()} | Components: {n_components}")

    # 1. Initialize & Train
    model = CSPClassifier(n_components=n_components, clf_type=clf_type)
    model.fit(X_train, y_train)

    # 2. Determine Evaluation Set
    if X_test is not None:
        print("[Status] Evaluating on TEST set...")
        eval_X, eval_y = X_test, y_test
    else:
        print("[Status] Evaluating on VALIDATION set...")
        eval_X, eval_y = X_val, y_val

    # 3. Evaluate
    preds = model.predict(eval_X)
    
    # Calculate Metrics
    bal_acc = balanced_accuracy_score(eval_y, preds)
    conf_matrix = confusion_matrix(eval_y, preds)
    report = classification_report(eval_y, preds, target_names=['Non-Target', 'Target'], zero_division=0)
    
    results = {
        'score': bal_acc,
        'report': report,
        'confusion_matrix': conf_matrix,
        'model': model  # <--- NEW: Return the trained model object!
    }

    print(f"Balanced Acc: {bal_acc:.4f}")
    return results