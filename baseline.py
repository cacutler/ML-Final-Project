import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
COLS = ["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15", "CANCELLED", "DIVERTED"]
SMOOTHING = 20
SEED = 6470
def load(data_dir, airports):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir}")
    df = pd.concat((pd.read_csv(f, usecols=COLS) for f in files), ignore_index=True)
    n0 = len(df)
    df = df[df["ORIGIN"].isin(airports)]
    n_airport = len(df)
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].dropna(subset=["ARR_DEL15"])
    print(f"Rows: {n0} loaded -> {n_airport} at chosen origins -> {len(df)} after dropping cancelled/diverted/missing target")
    df = df.assign(y=df["ARR_DEL15"].astype(int), date=pd.to_datetime(df["FL_DATE"]), dep_hour=(df["CRS_DEP_TIME"] // 100).clip(0, 23))
    df["route"] = df["ORIGIN"] + "-" + df["DEST"]
    df["origin_hour"] = df["ORIGIN"] + "_" + df["dep_hour"].astype(str)
    df["month"] = df["date"].dt.to_period("M")
    return df
def chrono_split(df):
    months = sorted(df["month"].unique())
    if len(months) < 3:
        raise SystemExit("Need at least 3 months (train / validation / test).")
    test_m, val_m = months[-1], months[-2]
    train = df[df["month"] < val_m]
    val = df[df["month"] == val_m]
    test = df[df["month"] == test_m]
    print(f"Train months: {months[:-2][0]}..{months[-3]} ({len(train)} rows) | Val: {val_m} ({len(val)}) | Test: {test_m} ({len(test)})")
    return train, val, test
def fit_rate_table(train, key, m=SMOOTHING):
    prior = train["y"].mean()
    g = train.groupby(key)["y"].agg(["sum", "count"])
    return ((g["sum"] + m * prior) / (g["count"] + m)), prior
def predict_rate(table_prior, df, key):
    table, prior = table_prior
    return df[key].map(table).fillna(prior).to_numpy()
def best_f1_threshold(y, scores):
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return t[int(np.argmax(f1))]
def evaluate(name, y_val, s_val, y_test, s_test, rows):
    row = {"model": name, "PR-AUC": average_precision_score(y_test, s_test), "ROC-AUC": roc_auc_score(y_test, s_test) if len(np.unique(s_test)) > 1 else 0.5}
    if len(np.unique(s_val)) > 1:
        thr = best_f1_threshold(y_val, s_val)  # chosen on validation, not test
        pred = (s_test >= thr).astype(int)
        row.update({"threshold": thr, "precision@thr": precision_score(y_test, pred, zero_division=0), "recall@thr": recall_score(y_test, pred, zero_division=0)})
    else:  # constant score: no meaningful threshold
        row.update({"threshold": np.nan, "precision@thr": np.nan, "recall@thr": 0.0})
    rows.append(row)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--airports", nargs="+", required=True)
    ap.add_argument("--out", default="baseline_results.csv")
    args = ap.parse_args()
    df = load(args.data_dir, args.airports)
    train, val, test = chrono_split(df)
    y_val, y_test = val["y"].to_numpy(), test["y"].to_numpy()
    print(f"\nDelay rate  train: {train['y'].mean():.3f}  val: {y_val.mean():.3f}  test: {y_test.mean():.3f}")
    print(f"Accuracy of 'always on time' on test: {1 - y_test.mean():.3f} (looks great, but recall is 0)\n")
    rows = []
    # 1. Always on time / constant score: PR-AUC collapses to the delay rate.
    evaluate("always_on_time", y_val, np.zeros(len(val)), y_test, np.zeros(len(test)), rows)
    # 2-4. Smoothed delay-rate lookups fit on TRAIN only.
    for name, key in [("carrier_rate", "OP_UNIQUE_CARRIER"), ("route_rate", "route"), ("origin_hour_rate", "origin_hour")]:
        tp = fit_rate_table(train, key)
        evaluate(name, y_val, predict_rate(tp, val, key), y_test, predict_rate(tp, test, key), rows)
    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False))
    res.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")
if __name__ == "__main__":
    main()