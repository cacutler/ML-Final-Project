from helpers import chrono_split, evaluate, load, SEED, MAX_ITERATIONS, CATEGORICAL, NUMERICAL, FEATURES
import argparse
from sklearn.ensemble import HistGradientBoostingClassifier
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./")
    ap.add_argument("--airports", nargs="+", required=True)
    ap.add_argument("--out", default="gradient_boosting_results.csv")
    args = ap.parse_args()
    df = load(args.data_dir, args.airports)
    train, val, test = chrono_split(df)
if __name__ == "__main__":
    main()