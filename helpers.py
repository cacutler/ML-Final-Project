import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, accuracy_score
TARGET = "ARR_DEL15"
# Columns every run needs for filtering, splitting, and the target.
REQUIRED_COLS = ["FL_DATE", "ORIGIN", "CANCELLED", "DIVERTED", TARGET]
# Known BEFORE departure: safe for the main models.
PRE_DEPARTURE_COLS = ["DAY_OF_WEEK", "OP_UNIQUE_CARRIER", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE"]
# Known only AFTER departure: use ONLY in the leakage experiment.
LEAKAGE_COLS = ["DEP_DELAY", "DEP_DEL15", "TAXI_OUT", "WHEELS_OFF"]
ALL_COLS = REQUIRED_COLS + PRE_DEPARTURE_COLS + LEAKAGE_COLS
CATEGORICAL = ["OP_UNIQUE_CARRIER", "DEST", "DAY_OF_WEEK", "dep_hour", "ORIGIN"]
NUMERICAL = ["CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE"]
FEATURES = CATEGORICAL + NUMERICAL
LEAKAGE_CATEGORICAL = ["DEP_DEL15"]
LEAKAGE_NUMERICAL = ["DEP_DELAY", "TAXI_OUT", "WHEELS_OFF"]
LEAKAGE_FEATURES = FEATURES + LEAKAGE_CATEGORICAL + LEAKAGE_NUMERICAL
SEED = 6470
MAX_ITERATIONS = 1000
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 50
#Claude generated some helper functions that I can use for all of the models I plan to create and use.
def load(data_dir, airports, cols=None):
    """Load monthly BTS CSVs, keep chosen origin airports, drop cancelled/diverted/missing target.
    cols: extra columns to read (default: every column used anywhere in the project).
    Adds: y (0/1 target), date, month, dep_hour (scheduled departure hour).
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir}")
    wanted = list(dict.fromkeys(REQUIRED_COLS + (cols if cols is not None else ALL_COLS)))
    df = pd.concat((pd.read_csv(f, usecols=wanted) for f in files), ignore_index=True)
    n0 = len(df)
    df = df[df["ORIGIN"].isin(airports)]
    n_airport = len(df)
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].dropna(subset=[TARGET])
    print(f"Rows: {n0} loaded -> {n_airport} at chosen origins -> {len(df)} after dropping cancelled/diverted/missing target")
    df = df.assign(y=df[TARGET].astype(int), date=pd.to_datetime(df["FL_DATE"]))
    df["month"] = df["date"].dt.to_period("M")
    if "CRS_DEP_TIME" in df:
        df["dep_hour"] = (df["CRS_DEP_TIME"] // 100).clip(0, 23)
    return df
def chrono_split(df):
    """Last month = test, second-last = validation, everything earlier = train."""
    months = sorted(df["month"].unique())
    if len(months) < 3:
        raise SystemExit("Need at least 3 months (train / validation / test).")
    test_m, val_m = months[-1], months[-2]
    train = df[df["month"] < val_m]
    val = df[df["month"] == val_m]
    test = df[df["month"] == test_m]
    print(f"Train months: {months[0]}..{months[-3]} ({len(train)} rows) | Val: {val_m} ({len(val)}) | Test: {test_m} ({len(test)})")
    return train, val, test
def best_f1_threshold(y, scores):
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return t[int(np.argmax(f1))]
def evaluate(name, y_val, s_val, y_test, s_test, rows):
    """Append one result row. Threshold is chosen on validation, applied once to test."""
    row = {"model": name, "PR-AUC": average_precision_score(y_test, s_test), "ROC-AUC": roc_auc_score(y_test, s_test) if len(np.unique(s_test)) > 1 else 0.5}
    if len(np.unique(s_val)) > 1:
        thr = best_f1_threshold(y_val, s_val)
        pred = (s_test >= thr).astype(int)
        row.update({"accuracy@thr": accuracy_score(y_test, pred), "threshold": thr, "precision@thr": precision_score(y_test, pred, zero_division=0), "recall@thr": recall_score(y_test, pred, zero_division=0)})
    else:  # constant score: no meaningful threshold
        row.update({"accuracy@thr": accuracy_score(y_test, np.zeros(len(y_test))), "threshold": np.nan, "precision@thr": np.nan, "recall@thr": 0.0})
    rows.append(row)