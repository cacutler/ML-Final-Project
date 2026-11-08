import argparse
import pandas as pd
from helpers import chrono_split, evaluate, load
SMOOTHING = 20  # pseudo-count pulling rare groups toward the global rate
#Claude helped me with getting the specific logic for the baseline model since I didn't know how to do it myself.
def add_baseline_keys(df):
    df = df.copy()
    df["route"] = df["ORIGIN"] + "-" + df["DEST"]
    df["origin_hour"] = df["ORIGIN"] + "_" + df["dep_hour"].astype(str)
    return df
def fit_rate_table(train, key, m=SMOOTHING):
    prior = train["y"].mean()
    g = train.groupby(key)["y"].agg(["sum", "count"])
    return ((g["sum"] + m * prior) / (g["count"] + m)), prior
def predict_rate(table_prior, df, key):
    table, prior = table_prior
    return df[key].map(table).fillna(prior).to_numpy()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./")
    ap.add_argument("--airports", nargs="+", required=True)
    ap.add_argument("--out", default="baseline_results.csv")
    args = ap.parse_args()
    df = add_baseline_keys(load(args.data_dir, args.airports))
    train, val, test = chrono_split(df)
    y_val, y_test = val["y"].to_numpy(), test["y"].to_numpy()
    print(f"\nDelay rate  train: {train['y'].mean():.3f}  val: {y_val.mean():.3f}  test: {y_test.mean():.3f}")
    print(f"Accuracy of 'always on time' on test: {1 - y_test.mean():.3f} (looks great, but recall is 0)\n")
    rows = []
    evaluate("always_on_time", y_val, [0] * len(val), y_test, [0] * len(test), rows)
    for name, key in [("carrier_rate", "OP_UNIQUE_CARRIER"), ("route_rate", "route"), ("origin_hour_rate", "origin_hour")]:
        tp = fit_rate_table(train, key)
        evaluate(name, y_val, predict_rate(tp, val, key), y_test, predict_rate(tp, test, key), rows)
    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False))
if __name__ == "__main__":
    main()