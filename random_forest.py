from helpers import chrono_split, evaluate, load, PRE_DEPARTURE_COLS, LEAKAGE_COLS, SEED
import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./")
    ap.add_argument("--airports", nargs="+", required=True)
    ap.add_argument("--out", default="random_forest_results.csv")
    args = ap.parse_args()
    df = load(args.data_dir, args.airports)
    train, val, test = chrono_split(df)
    features = PRE_DEPARTURE_COLS
    features_leaky = PRE_DEPARTURE_COLS + LEAKAGE_COLS
if __name__ == "__main__":
    main()