from helpers import chrono_split, evaluate, load, SEED, LEAKAGE_CATEGORICAL, LEAKAGE_NUMERICAL, LEAKAGE_FEATURES, N_ESTIMATORS, MIN_SAMPLES_LEAF
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./")
    ap.add_argument("--airports", nargs="+", required=True)
    ap.add_argument("--balanced", action="store_true", help="use class_weight='balanced' (default: no class weights)")
    ap.add_argument("--out", default="random_forest_results.csv")
    args = ap.parse_args()
    df = load(args.data_dir, args.airports)
    train, val, test = chrono_split(df)
    preprocessor = ColumnTransformer(transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), LEAKAGE_NUMERICAL), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), LEAKAGE_CATEGORICAL)])#Claude pointed out that I should use the OrdinalEncoder rather than the OneHotEncoder for the random forest.
    model = RandomForestClassifier(class_weight="balanced_subsample" if args.balanced else None, n_jobs=-1, min_samples_leaf=MIN_SAMPLES_LEAF, n_estimators=N_ESTIMATORS, random_state=SEED)#Claude pointed out a few changes to the random forest model such as the min_samples_leaf and the n_jobs parameters.
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(train[LEAKAGE_FEATURES], train["y"])
    s_val = pipeline.predict_proba(val[LEAKAGE_FEATURES])[:, 1]
    s_test = pipeline.predict_proba(test[LEAKAGE_FEATURES])[:, 1]
    rows = []
    name = "randomforest_balanced" if args.balanced else "randomforest"
    evaluate(name, val["y"], s_val, test["y"], s_test, rows)
    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False))
if __name__ == "__main__":
    main()