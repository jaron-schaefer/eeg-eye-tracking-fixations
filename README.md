# Eye‑EEG: Target vs. Non‑Target Fixation Classification

This work uses the public EEGET‑RSOD dataset from the paper “A simultaneous EEG and eye‑tracking dataset for remote sensing object detection,” reproducing and extending its core analysis: predicting whether a fixation landed on a target (airplane).

## Summary
This repository provides a compact pipeline: load raw EEG/ET, align and label fixations, extract fixation‑locked EEG epochs, baseline‑correct, and optionally engineer features. We compare several models (EEGNet, CSP, CBraMod, Random Forest on features) with subject‑wise splits to avoid leakage. The goal is a clear, end‑to‑end reference for target vs. non‑target fixation classification on remote‑sensing imagery.

## Data
We use the EEGET‑RSOD dataset (“A simultaneous EEG and eye‑tracking dataset for remote sensing object detection”). Place its contents under `data/raw_data/` (e.g., `EEG_clean/`, `ET_clip/`, `Stimuli/`, `label/`). EEG is converted from volts to microvolts (μV) on load; epochs are baseline‑corrected before saving.

## Pipeline
Run the preparation script to label fixations and create training data. It produces per‑subject NPZs in `data/labeled_eeg_data/` (epochs `X`, labels `y`, `time`, `channel_names`, `sampling_rate`, and a `stimulus_summary` per fixation) and, if enabled, feature CSVs under `data/features/`.


Training lives in the `classification/` notebooks. Use `classification/training_split.py` for randomized or leave‑N‑out subject‑wise splits. Adjust channels, subjects, and CV in `data/data_configs.py`.
