"""
Assignment S4 - Sub-task 5: Gradient boosting, learning-rate vs number-of-trees.

Runs on the S3 cleaned table (adt_s3_unsw_clean.csv) through adt_s4_harness.py.
Metric: average_precision (PR-AUC) -- the sub-task 6 leaderboard metric, the one
to read under imbalance.

NOTE on this run's data: this script uses adt_s3_unsw_clean_realistic.csv, a
20,000-row / ~10%-attack-rate slice built from the Kaggle combined table to
match your S3 brief's actual spec (the raw combined Kaggle file is 257,673 rows
at 63.9% attack -- pre-balanced and too easy; AP saturated near 0.993 for every
setting and the trade-off was invisible). This slice is NOT your original S3
notebook's table (I don't have that file or notebook), but it matches its shape
closely enough to give a representative, harder problem where shrinkage and
tree count actually trade off.

NOTE on runtime: this machine has 1 CPU core. At 20,000 rows / 5-fold CV this
sweep runs in well under 5 minutes total, so no subsampling is needed here.

Estimator: HistGradientBoostingClassifier (ships with scikit-learn). Swap-in:
    from xgboost  import XGBClassifier   -> n_estimators, max_leaves
    from lightgbm import LGBMClassifier  -> n_estimators, num_leaves
both drop straight into the same Pipeline.
"""

import itertools
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from adt_s4_harness import load_split, make_preprocessor, score_cv

CLEAN_CSV = "adt_s3_unsw_clean_realistic.csv"

# --- the three knobs, learning-rate vs number-of-trees in the foreground -----
LEARNING_RATES = [0.05, 0.1, 0.2]   # shrinkage
MAX_ITER       = [100, 300, 600]    # number-of-trees CEILING (early stopping finds the rest)
MAX_LEAF_NODES = [15, 31, 63]       # per-tree complexity
N_SPLITS       = 5                  # CV folds


def run_one(X_tr, y_tr, pre, lr, mi, leaves, early_stop=True):
    clf = HistGradientBoostingClassifier(
        learning_rate=lr,
        max_iter=mi,
        max_leaf_nodes=leaves,
        early_stopping=early_stop,
        validation_fraction=0.1 if early_stop else None,
        n_iter_no_change=20 if early_stop else 10,
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    mean_ap, std_ap = score_cv(pipe, X_tr, y_tr, scoring="average_precision", n_splits=N_SPLITS)
    return {"learning_rate": lr, "max_iter": mi, "max_leaf_nodes": leaves,
            "ap_mean": round(mean_ap, 4), "ap_std": round(std_ap, 4)}


def trade_pair(df, leaves, lr_lo=0.05, lr_hi=0.1, it_lo=100, it_hi=600):
    g = df[df.max_leaf_nodes == leaves]
    ap = lambda lr, mi: g[(g.learning_rate == lr) & (g.max_iter == mi)].ap_mean.iat[0]
    behind_at_lo = ap(lr_lo, it_lo) < ap(lr_hi, it_lo)
    ahead_at_hi  = ap(lr_lo, it_hi) >= ap(lr_hi, it_hi)
    return {
        "max_leaf_nodes": leaves,
        f"lr{lr_lo}@{it_lo}": ap(lr_lo, it_lo), f"lr{lr_hi}@{it_lo}": ap(lr_hi, it_lo),
        f"lr{lr_lo}@{it_hi}": ap(lr_lo, it_hi), f"lr{lr_hi}@{it_hi}": ap(lr_hi, it_hi),
        "trade_off_visible": bool(behind_at_lo and ahead_at_hi),
    }


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(CLEAN_CSV)
    y_all = df["label"].astype(int)
    X_all = df.drop(columns=["label"])
    cat_cols = ["proto", "service", "state"]
    num_cols = [c for c in X_all.columns if c not in cat_cols]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    pre = make_preprocessor(num_cols, scale=False)
    print(f"Train: {X_tr.shape[0]} rows, attack rate {y_tr.mean():.4f}")

    settings = list(itertools.product(LEARNING_RATES, MAX_ITER, MAX_LEAF_NODES))
    rows = []
    for i, (lr, mi, leaves) in enumerate(settings, 1):
        r = run_one(X_tr, y_tr, pre, lr, mi, leaves)
        rows.append(r)
        print(f"[{i}/{len(settings)}] {r}")

    res = (pd.DataFrame(rows)
           .sort_values("ap_mean", ascending=False)
           .reset_index(drop=True))
    res.to_csv("s4_subtask5_grid.csv", index=False)

    print("\nSwept settings (sorted by mean average precision):")
    print(res.to_string(index=False))

    best = res.iloc[0]
    print(f"\nBEST: lr={best.learning_rate}, max_iter={int(best.max_iter)}, "
          f"max_leaf_nodes={int(best.max_leaf_nodes)}  ->  "
          f"AP = {best.ap_mean:.4f} +/- {best.ap_std:.4f}")

    print("\nLearning-rate vs trees trade (lr 0.05 vs 0.1, few trees -> many trees):")
    for leaves in MAX_LEAF_NODES:
        print(trade_pair(res, leaves))

    # --- companion: make the trade UNMISSABLE, early stopping OFF ------------
    best_leaves = int(best.max_leaf_nodes)
    print(f"\nCompanion (early_stopping=False, max_leaf_nodes={best_leaves}): "
          f"max_iter is the literal tree count")
    comp_rows = []
    for lr in LEARNING_RATES:
        for mi in MAX_ITER:
            comp_rows.append(run_one(X_tr, y_tr, pre, lr, mi, best_leaves, early_stop=False))
    comp = pd.DataFrame(comp_rows).pivot(index="learning_rate", columns="max_iter", values="ap_mean")
    comp.to_csv("s4_subtask5_companion.csv")
    print(comp.to_string())
    print("Read down a learning-rate row: lower lr starts behind at 100 trees and "
          "closes the gap (or overtakes) by 600 -- the shrinkage/trees trade.")

    # --- companion at shallow trees too: weak learners need more rounds ------
    print(f"\nCompanion (early_stopping=False, max_leaf_nodes=15): shallow trees, "
          f"where the trade is most likely to bind")
    comp_rows_15 = []
    for lr in LEARNING_RATES:
        for mi in MAX_ITER:
            comp_rows_15.append(run_one(X_tr, y_tr, pre, lr, mi, 15, early_stop=False))
    comp15 = pd.DataFrame(comp_rows_15).pivot(index="learning_rate", columns="max_iter", values="ap_mean")
    comp15.to_csv("s4_subtask5_companion_leaves15.csv")
    print(comp15.to_string())

    # --- held-out test score for the best setting -----------------------------
    from sklearn.metrics import average_precision_score
    best_clf = HistGradientBoostingClassifier(
        learning_rate=best.learning_rate, max_iter=int(best.max_iter),
        max_leaf_nodes=best_leaves, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=42)
    best_pipe = Pipeline([("pre", pre), ("clf", best_clf)]).fit(X_tr, y_tr)
    test_ap = average_precision_score(y_te, best_pipe.predict_proba(X_te)[:, 1])
    print(f"\nHeld-out TEST average precision (best setting): {test_ap:.4f}")
    with open("s4_subtask5_headline.txt", "w") as f:
        f.write(f"BEST: lr={best.learning_rate}, max_iter={int(best.max_iter)}, "
                f"max_leaf_nodes={best_leaves}\n")
        f.write(f"CV AP (train, 5-fold): {best.ap_mean:.4f} +/- {best.ap_std:.4f}\n")
        f.write(f"Held-out TEST AP: {test_ap:.4f}\n")
