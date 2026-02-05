
from classification.training_utils import optimize_single_model
from data.data_configs import SUBJECTS

PARAM_GRIDS = {
    'eegnet': {
        'D': [2, 4],
        'F1': [8, 16],
        'F2': [16],
        'dropoutRate': [0.3],
        'lr': [0.001, 0.0005],
    },
    'cbramod':{
        'dropout': [0.1],
        'hidden_dim': [200, 256, 512],
        'weight_decay': [0.05],
        'lr': [ 0.0001, 0.00005],
        'label_smoothing': [0.0]
    },
    'csp': {
        'n_components': [6, 10, 15],
        'reg': [0.01, 0.05, 0.1],
        'clf_type': ['lda', 'rf']
    },
    'features_rf': {
        'n_estimators': [100, 200],
        'max_depth': [20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'features_svm': {
        'C': [0.01, 0.1, 1.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    },
    'et_rf': {
        'n_estimators': [50, 100, 200],
        'max_depth': [20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'et_svm': {
        'C': [0.01, 0.1, 1.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

for model in PARAM_GRIDS.keys():
    print(f"Optimizing {model}...")
    optimize_single_model(model, SUBJECTS, PARAM_GRIDS[model], max_trials=30)