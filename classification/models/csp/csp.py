from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

class CSPClassifier:
    """
    Standard BCI Pipeline:
    CSP (Spatial Filter) -> Feature Scaling -> Classifier (LDA or RF)
    """
    def __init__(self, n_components=4, clf_type='lda', seed=42):
        self.n_components = n_components
        self.clf_type = clf_type
        self.seed = seed
        self.model = self._build_pipeline()

    def _build_pipeline(self):
        # 1. CSP (Spatial Filtering)
        csp = CSP(n_components=self.n_components, 
                  reg=None, 
                  log=True,         # Log-variance is the standard feature
                  norm_trace=False)

        # 2. Feature Scaling (StandardScaler)
        # Note: We scale the FEATURES, not the raw data.
        scaler = StandardScaler()

        # 3. Classifier
        if self.clf_type == 'lda':
            # Fast, robust, prevents overfitting
            clf = LinearDiscriminantAnalysis()
            
        elif self.clf_type == 'rf':
            # Regularized Random Forest (Depth limited to 5)
            clf = RandomForestClassifier(
                n_estimators=100, 
                max_depth=5,        # Limit depth to stop memorization
                max_features='sqrt',
                random_state=self.seed,
                n_jobs=-1
            )
        else:
            raise ValueError("clf_type must be 'lda' or 'rf'")

        return Pipeline([
            ('csp', csp),
            ('scaler', scaler),
            ('clf', clf)
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)