# EEG Classification Module

## Model Optimization Pipeline

Clean and simple pipeline to optimize and evaluate all EEG classification models.

### Quick Start

```bash
# Local usage
python run_optimization.py optimize    # Find optimal parameters
python run_optimization.py evaluate    # Evaluate with optimal parameters
python run_optimization.py quick       # Quick test run
```

### Google Colab (Recommended for GPU)

Upload `model_optimization_colab.ipynb` to Google Colab for GPU-accelerated optimization.

### Files

- `model_optimization.py`: Main optimization functions
- `run_optimization.py`: Simple command-line runner  
- `model_optimization_colab.ipynb`: Google Colab notebook with GPU support

### Supported Models

- **EEGNet**: Deep learning for raw EEG
- **CBRAMod**: Cross-attention transformer for raw EEG  
- **CSP**: Common Spatial Patterns for raw EEG
- **Feature RF/SVM**: Random Forest/SVM on EEG features
- **ET RF/SVM**: Random Forest/SVM on eye tracking features

### Output

Results saved to `optimization_results/`:
- `optimal_parameters_latest.json`: Best parameters for each model
- `evaluation_results_latest.json`: Final test performance
- Timestamped versions with full history

### Channel Groups (for future analysis)

- **Frontal**: ['F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2']
- **Central**: ['FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4'] 
- **Parietal**: ['P3', 'P4', 'P7', 'P8', 'Pz']
- **Occipital**: ['O1', 'O2', 'Oz']
- **Temporal**: ['T7', 'T8', 'TP9', 'TP10']

---

## Individual Classifiers

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