# adt_s4_harness.py
# Thin starter harness for the S4 assignment. Three helpers: load_split,
# make_preprocessor, score_cv. Import them and every sub-task is fit, score, read.

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

CLEAN_CSV = "adt_s3_unsw_clean.csv"        # written by the S3 cleaning notebook
CAT_COLS  = ["proto", "service", "state"]  # the three text columns


def load_split(path=CLEAN_CSV, test_size=0.2, seed=42):
    """Return X_train, X_test, y_train, y_test, num_cols, cat_cols.
    Stratified, so the attack share holds in both splits."""
    df = pd.read_csv(path)
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    return X_tr, X_te, y_tr, y_te, num_cols, CAT_COLS


def make_preprocessor(num_cols, cat_cols=CAT_COLS, scale=True):
    """Shared leak-free preprocessing: standardise the numeric columns and
    one-hot encode the three categoricals. Put it inside a Pipeline so it
    refits on each fold's training rows only. Set scale=False for tree models,
    which do not need standardisation. sparse_output=False keeps the matrix
    dense, which GaussianNB and a few displays need."""
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num_step, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def score_cv(estimator, X, y, scoring="roc_auc", n_splits=5, seed=42):
    """Leak-free cross-validated score. Pass a Pipeline (preprocessing plus
    estimator) so nothing leaks across the fold boundary. Returns the mean and
    the standard deviation of the per-fold scores. The default metric is the
    Area Under the Curve (roc_auc), which is fine for tuning, but for the
    sub-task 6 leaderboard pass scoring="average_precision", because the data is
    imbalanced and average precision is the metric S3 told you to read under
    imbalance."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    s = cross_val_score(estimator, X, y, scoring=scoring, cv=cv)
    return s.mean(), s.std()
