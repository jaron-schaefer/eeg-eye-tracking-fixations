# EEG-Eye Tracking Classification Project

## Abstract

This study investigates the classification of eye fixations on target objects (airplanes) in remote sensing imagery using synchronized EEG and eye-tracking signals from the EEGET-RSOD dataset introduced by He et al. (2025). We implement and compare multiple machine learning approaches including deep learning models (EEGNet, CBraMod) and traditional methods (CSP, Random Forest) to predict whether a fixation landed on a target or non-target region.

## Dataset

The project uses the [**EEGET-RSOD dataset**](https://figshare.com/articles/dataset/EEGET-RSOD/26943565), which contains:
- EEG signals recorded during visual search tasks
- Eye-tracking data with fixation coordinates
- Remote sensing images with airplane targets
- Synchronized data from multiple participants


## Methodology

The pipeline includes:

1. **Data Preprocessing**: EEG signal filtering, eye-tracking data alignment, and fixation-target labeling
2. **Data Quality Assurance**: Notebooks to check time alignment, ET calibration quality, and selection of best subjects
3. **Feature Extraction**: Time-domain and frequency-domain EEG features, fixation-locked epochs
4. **Classification Models**:
   - **EEGNet**: Deep learning architecture specialized for EEG classification
   - **CBraMod**: Transformer-based EEG classifier (pretrained)
   - **CSP**: Common Spatial Patterns with traditional classifiers
   - **Feature-based Classifier**: Engineered EEG and eye-tracking features with Random Forest
5. **Hyperparameter Optimization**: Automated tuning for each model architecture
6. **Evaluation**: Performance assessment using both pooled and cross-subject validation

## Quick Start

### Installation
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation
Download and preprocess the EEGET-RSOD dataset. This step handles EEG signal filtering, eye-tracking alignment, and fixation labeling.
> **Note**: This step downloads the complete dataset (~several GB) and processes all EEG/eye-tracking files, which may take 90+ minutes depending on your internet connection and system performance.

```bash
python -m data.run_data_preparation
```

### Model Optimization
Run hyperparameter optimization to find the best parameters for each classification model (EEGNet, CBraMod, CSP, Random Forest):
```bash
python -m classification.run_optimization
```

### Model Evaluation
Evaluate the optimized models using both pooled and cross-subject validation to assess performance and generalizability:
```bash
python -m classification.run_evaluation
```

## Results

Optimization and classification results are automatically saved to `classification/results/` after running the evaluation pipeline.