from helpers import chrono_split, evaluate, load, SEED, MAX_ITERATIONS, LEAKAGE_CATEGORICAL, LEAKAGE_NUMERICAL, LEAKAGE_FEATURES
import argparse
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./")
    ap.add_argument("--airports", nargs="+", required=True)
    ap.add_argument("--balanced", action="store_true", help="use class_weight='balanced' (default: no class weights)")
    ap.add_argument("--out", default="gradient_boosting_results.csv")
    args = ap.parse_args()
    df = load(args.data_dir, args.airports)
    train, val, test = chrono_split(df)
    preprocessor = ColumnTransformer(transformers=[("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), LEAKAGE_CATEGORICAL), ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), LEAKAGE_NUMERICAL)])#Claude helped me rework the preprocessor pipeline for the hist gradient boosting model.
    model = HistGradientBoostingClassifier(categorical_features=list(range(len(LEAKAGE_CATEGORICAL))), early_stopping=False, max_iter=MAX_ITERATIONS, random_state=SEED)#Claude helped me in pointing out the parameters I should use for the hist gradient boosting classifier.
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(train[LEAKAGE_FEATURES], train["y"])
    s_val = pipeline.predict_proba(val[LEAKAGE_FEATURES])[:, 1]
    s_test = pipeline.predict_proba(test[LEAKAGE_FEATURES])[:, 1]
    rows = []
    name = "histgradientboosting_balanced_leakage" if args.balanced else "histgradientboosting_leakage"
    evaluate(name, val["y"], s_val, test["y"], s_test, rows)
    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False))
if __name__ == "__main__":
    main()