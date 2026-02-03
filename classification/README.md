# EEG Classification Module

This module provides three integrated EEG classifiers with built-in data preparation, training, and optimization capabilities.

## Classifiers

### 1. CSP Classifier (`classification/models/csp/csp_classifier.py`)
- **Method**: Common Spatial Patterns with LDA/SVM/Random Forest
- **Features**: Bandpass filtering (0.5-40 Hz), spatial filtering
- **Best for**: Motor imagery tasks, simple feature extraction
- **Usage**:
```python
from classification.models.csp.csp_classifier import CSPClassifier

csp = CSPClassifier(classifier_type='lda')
results = csp.train(X_train, y_train, X_val, y_val, X_test, y_test)
predictions = csp.predict(X_new)
```

### 2. EEGNet Classifier (`classification/models/eeg_net/eegnet_classifier.py`)
- **Method**: Depthwise separable CNN 
- **Features**: Per-subject normalization, early stopping
- **Best for**: General EEG classification, automatic feature learning
- **Usage**:
```python
from classification.models.eeg_net.eegnet_classifier import EEGNetClassifier

eegnet = EEGNetClassifier()
results = eegnet.train(X_train, y_train, X_val, y_val, X_test, y_test)
predictions = eegnet.predict(X_new)
```

### 3. CBraMod Classifier (`classification/models/cbramod/cbramod_classifier.py`)
- **Method**: Pretrained transformer foundation model
- **Features**: 200Hz resampling, artifact removal, label smoothing
- **Best for**: Complex temporal patterns, transfer learning
- **Usage**:
```python
from classification.models.cbramod.cbramod_classifier import CBraModClassifier

cbramod = CBraModClassifier()
results = cbramod.train(X_train, y_train, X_val, y_val, X_test, y_test)
predictions = cbramod.predict(X_new)
```

## Key Features

### Integrated Data Preparation
- Each classifier handles its own data preprocessing
- Automatic resampling, filtering, and normalization
- Artifact removal and validation

### Built-in Training
- Optimized hyperparameters for each method
- Early stopping and regularization
- GPU acceleration when available

### Parameter Optimization
- Grid search for hyperparameter tuning
- Balanced accuracy optimization
- Cross-validation support

## Quick Start

```python
# Load your data
X_train, y_train, X_val, y_val, X_test, y_test = load_your_data()

# Try all classifiers
from classification.models.csp.csp_classifier import CSPClassifier
from classification.models.eeg_net.eegnet_classifier import EEGNetClassifier  
from classification.models.cbramod.cbramod_classifier import CBraModClassifier

# CSP
csp = CSPClassifier()
csp_results = csp.train(X_train, y_train, X_val, y_val, X_test, y_test)

# EEGNet
eegnet = EEGNetClassifier()
eegnet_results = eegnet.train(X_train, y_train, X_val, y_val, X_test, y_test)

# CBraMod  
cbramod = CBraModClassifier()
cbramod_results = cbramod.train(X_train, y_train, X_val, y_val, X_test, y_test)

print(f"CSP: {csp_results['balanced_acc']:.3f}")
print(f"EEGNet: {eegnet_results['balanced_acc']:.3f}") 
print(f"CBraMod: {cbramod_results['balanced_acc']:.3f}")
```

## Data Format

All classifiers expect data in the format:
- **X**: `(n_samples, n_channels, n_timepoints)` - EEG data
- **y**: `(n_samples,)` - Binary labels (0/1)
- **subject_ids**: `(n_samples,)` - Subject identifiers (optional)

## Requirements

- PyTorch (for CBraMod)
- TensorFlow (for EEGNet) 
- scikit-learn (for CSP)
- MNE (for preprocessing)
- NumPy, SciPy

## Run Demo

```bash
python classification/demo_classifiers.py
```