from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier

from data.data_configs import FEATURES_DIR, SUBJECTS


def _subject_features_path(subject: str) -> Path:
	return Path(FEATURES_DIR) / subject / "features.csv"


def load_feature_dataset(subjects: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
	"""
	Load and concatenate per-subject feature CSVs into X, y.

	Returns
	- X: (n_samples, n_features)
	- y: (n_samples,)
	- columns: feature column names in X
	"""
	dfs: List[pd.DataFrame] = []
	for s in subjects:
		fp = _subject_features_path(s)
		if not fp.exists():
			print(f"[skip] Missing features for {s} at {fp}")
			continue
		df = pd.read_csv(fp)
		if "label" not in df.columns:
			print(f"[skip] No label column in {fp}")
			continue
		dfs.append(df)

	if not dfs:
		raise FileNotFoundError("No feature CSVs found for provided subjects.")

	df_all = pd.concat(dfs, axis=0, ignore_index=True)
	y = df_all["label"].to_numpy().astype(int)
	X = df_all.drop(columns=["label"]).to_numpy(dtype=np.float32)
	columns = [c for c in df_all.columns if c != "label"]
	return X, y, columns


def build_model(model_type: str = "hgb") -> Pipeline:
	"""
	Build a simple classifier pipeline.

	- "svm": StandardScaler + LinearSVC
	- "hgb": HistGradientBoostingClassifier (no scaling needed)
	"""
	model_type = model_type.lower()
	if model_type == "svm":
		clf = Pipeline([
			("scaler", StandardScaler(with_mean=True, with_std=True)),
			("svm", LinearSVC(C=1.0, max_iter=5000))
		])
	elif model_type == "hgb":
		clf = HistGradientBoostingClassifier(
			learning_rate=0.1,
			max_depth=None,
			max_iter=300,
			random_state=42
		)
	else:
		raise ValueError("model_type must be 'svm' or 'hgb'")
	return clf


def train_and_evaluate(subjects: List[str] | None = None, model_type: str = "hgb") -> None:
	subjects = subjects or SUBJECTS
	X, y, cols = load_feature_dataset(subjects)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, stratify=y, random_state=42
	)

	clf = build_model(model_type)
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"Test accuracy: {acc:.4f}")
	print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
	print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Train a simple classifier on extracted EEG features.")
	parser.add_argument("--model", choices=["svm", "hgb"], default="hgb", help="Classifier type")
	parser.add_argument("--subjects", nargs="*", default=None, help="Subset of subjects to use (e.g., P01 P02)")
	args = parser.parse_args()

	train_and_evaluate(args.subjects, args.model)

