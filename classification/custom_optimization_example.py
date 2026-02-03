"""
Example of using custom parameter grids for optimization
"""

from classification.models.csp.csp_classifier import CSPClassifier
from classification.models.eeg_net.eegnet_classifier import EEGNetClassifier
from classification.models.cbramod.cbramod_classifier import CBraModClassifier

def custom_optimization_example():
    """Example showing how to use custom parameter grids"""
    
    # Load your data (replace with actual data loading)
    # X_train, y_train, X_val, y_val = load_your_data()
    
    # 1. CSP with custom parameter grid
    csp_param_grid = [
        # Test different CSP components and LDA regularization
        {'n_components': 4, 'clf_type': 'lda', 'reg': 'auto'},
        {'n_components': 6, 'clf_type': 'lda', 'reg': None},
        {'n_components': 8, 'clf_type': 'lda', 'reg': 0.1},
        
        # Test Random Forest with different depths and estimators
        {'n_components': 4, 'clf_type': 'rf', 'max_depth': 3, 'n_estimators': 50},
        {'n_components': 6, 'clf_type': 'rf', 'max_depth': 5, 'n_estimators': 100},
        {'n_components': 8, 'clf_type': 'rf', 'max_depth': 7, 'n_estimators': 200},
    ]
    
    csp = CSPClassifier()
    # csp_results = csp.optimize(X_train, y_train, X_val, y_val, param_grid=csp_param_grid)
    
    # 2. EEGNet with custom architecture parameters
    eegnet_param_grid = [
        # Test different CNN architectures
        {'dropoutRate': 0.3, 'F1': 4, 'D': 2, 'F2': 8, 'lr': 0.001, 'batch_size': 32},
        {'dropoutRate': 0.5, 'F1': 8, 'D': 2, 'F2': 16, 'lr': 0.001, 'batch_size': 32},
        {'dropoutRate': 0.5, 'F1': 16, 'D': 4, 'F2': 32, 'lr': 0.0001, 'batch_size': 64},
        
        # Test different learning rates and epochs
        {'dropoutRate': 0.5, 'F1': 8, 'D': 2, 'F2': 16, 'lr': 0.01, 'epochs': 50},
        {'dropoutRate': 0.7, 'F1': 8, 'D': 2, 'F2': 16, 'lr': 0.001, 'epochs': 100},
    ]
    
    eegnet = EEGNetClassifier()
    # eegnet_results = eegnet.optimize(X_train, y_train, X_val, y_val, param_grid=eegnet_param_grid)
    
    # 3. CBraMod with custom transformer parameters
    cbramod_param_grid = [
        # Test different dropout and learning rates
        {'dropout': 0.05, 'lr': 1e-5, 'batch_size': 4, 'epochs': 20},
        {'dropout': 0.1, 'lr': 5e-5, 'batch_size': 8, 'epochs': 30},
        {'dropout': 0.2, 'lr': 1e-4, 'batch_size': 16, 'epochs': 15},
        
        # Test longer training with smaller learning rates
        {'dropout': 0.1, 'lr': 1e-6, 'batch_size': 4, 'epochs': 50},
        {'dropout': 0.15, 'lr': 2e-5, 'batch_size': 8, 'epochs': 40},
    ]
    
    cbramod = CBraModClassifier()
    # cbramod_results = cbramod.optimize(X_train, y_train, X_val, y_val, param_grid=cbramod_param_grid)
    
    print("Custom optimization examples created!")
    print("\\nTo use:")
    print("1. Load your data")
    print("2. Uncomment the optimization lines")
    print("3. Run the optimization with your custom parameter grids")

if __name__ == "__main__":
    custom_optimization_example()