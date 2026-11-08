from helpers import chrono_split, evaluate, load, SEED, MAX_ITERATIONS, LEAKAGE_CATEGORICAL, LEAKAGE_NUMERICAL, LEAKAGE_FEATURES
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
def main():#Claude generated some psuedocode for me that I used to implement the pipeline and model training.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./")
    ap.add_argument("--airports", nargs="+", required=True)
    ap.add_argument("--balanced", action="store_true", help="use class_weight='balanced' (default: no class weights)")#Claude added this balanced flag
    ap.add_argument("--out", default="logistic_regression_results.csv")
    args = ap.parse_args()
    df = load(args.data_dir, args.airports)
    train, val, test = chrono_split(df)
    preprocessor = ColumnTransformer(transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), LEAKAGE_NUMERICAL), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), LEAKAGE_CATEGORICAL)])#Claude fixed some errors I had with my preprocessor.
    model = LogisticRegression(class_weight="balanced" if args.balanced else None, max_iter=MAX_ITERATIONS, random_state=SEED)
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(train[LEAKAGE_FEATURES], train["y"])
    s_val = pipeline.predict_proba(val[LEAKAGE_FEATURES])[:, 1]
    s_test = pipeline.predict_proba(test[LEAKAGE_FEATURES])[:, 1]
    rows = []
    name = "logreg_balanced" if args.balanced else "logreg"
    evaluate(name, val["y"], s_val, test["y"], s_test, rows)
    res = pd.DataFrame(rows).round(4)#Claude helped me write the results to the terminal and to a csv file.
    print(res.to_string(index=False))
if __name__ == "__main__":
    main()