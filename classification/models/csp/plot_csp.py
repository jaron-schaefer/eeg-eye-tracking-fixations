import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_configs import *

def get_csp_object_from_pipeline(model):
    """Helper to extract the raw CSP object from your Pipeline class."""
    # If it's your custom class, access the internal sklearn pipeline
    if hasattr(model, 'model'):
        pipeline = model.model
    else:
        pipeline = model
        
    # Find the step named 'csp'
    try:
        return pipeline.named_steps['csp']
    except (AttributeError, KeyError):
        raise ValueError("Could not find 'csp' step in the model pipeline.")

def plot_csp_patterns(model, ch_names, sfreq=EEG_SAMPLE_FREQUENCY, layout='standard_1020'):
    """
    Plots the Topomaps (Spatial Patterns) learned by CSP.
    Blue/Red areas show which electrodes are most important for discrimination.
    """
    csp = get_csp_object_from_pipeline(model)
    
    # Create Dummy Info for MNE Plotting
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Apply Montage (Location of sensors)
    try:
        montage = mne.channels.make_standard_montage(layout)
        info.set_montage(montage, on_missing='ignore')
    except ValueError:
        print(f"Warning: Montage '{layout}' not found. Plotting sensors as circles.")
    
    print("\n" + "="*60)
    print(" CSP SPATIAL PATTERNS (Topomaps)")
    print("="*60)
    print(" -> Pattern 1 (Top Left): Maximizes variance for Class 0")
    print(" -> Last Pattern (Bottom Right): Maximizes variance for Class 1")
    
    # MNE's built-in plotting function
    # 'patterns_' attribute contains the spatial filters inverse
    csp.plot_patterns(info, ch_type='eeg', units='AU', size=1.5, show_names=False)
    plt.show()

def plot_csp_feature_distributions(model, X_raw, y, n_plot=3):
    """
    Transforms the raw data using the trained CSP and plots histograms 
    of the resulting features.
    
    Shows how well the CSP features separate the two classes (Blue vs Red).
    """
    csp = get_csp_object_from_pipeline(model)
    
    # Transform raw data to get the CSP features
    # Note: We only run the 'csp' step, not the whole pipeline
    X_features = csp.transform(X_raw)
    
    n_components = X_features.shape[1]
    n_plot = min(n_plot, n_components // 2) # Plot pairs (first N and last N)
    
    print("\n" + "="*60)
    print(f" CSP FEATURE DISTRIBUTIONS (Separability)")
    print("="*60)
    
    print(" -> Ideally, Blue (Class 0) and Red (Class 1) distributions should be separate.")
    
    fig, axes = plt.subplots(n_plot, 2, figsize=(12, 4 * n_plot))
    if n_plot == 1: axes = axes.reshape(1, -1)
    
    # Indices of components to plot: First N and Last N
    # First N correspond to Class 0 variance, Last N to Class 1 variance
    indices_to_plot = list(range(n_plot)) + list(range(n_components - n_plot, n_components))
    
    colors = ['blue', 'red']
    labels = ['Non-Target', 'Target']
    
    for i, ax in enumerate(axes.flatten()):
        comp_idx = indices_to_plot[i]
        
        # Plot distribution for each class
        for class_idx in [0, 1]:
            # Extract features for this class
            feats = X_features[y == class_idx, comp_idx]
            
            sns.histplot(feats, color=colors[class_idx], label=labels[class_idx], 
                         kde=True, stat="density", alpha=0.4, ax=ax)
            
        ax.set_title(f"CSP Component #{comp_idx + 1}")
        ax.set_xlabel("Feature Value (Log-Variance)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()

def visualize_csp_analysis(model, X_train, y_train, ch_names=DEFAULT_EEG_CHANNELS , sfreq=EEG_SAMPLE_FREQUENCY):
    """
    Master function to run both visualizations.
    """
    # 1. Plot Topomaps (Patterns)
    plot_csp_patterns(model, ch_names, sfreq)
    
    # 2. Plot Feature Histograms (Separability)
    # We use X_train to see how well the model learned to separate the training data
    plot_csp_feature_distributions(model, X_train, y_train)