from helpers import chrono_split, evaluate, load, best_f1_threshold, SEED, MAX_ITERATIONS, CATEGORICAL, NUMERICAL, FEATURES
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
def main():#Claude generated some psuedocode for me that I used to implement the pipeline and model training.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./")
    ap.add_argument("--airports", nargs="+", required=True)
    ap.add_argument("--balanced", action="store_true", help="use class_weight='balanced' (default: no class weights)")#Claude added this balanced flag
    ap.add_argument("--out", default="logistic_regression_results.csv")
    args = ap.parse_args()
    df = load(args.data_dir, args.airports)
    train, val, test = chrono_split(df)
    preprocessor = ColumnTransformer(transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERICAL), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)])#Claude fixed some errors I had with my preprocessor.
    model = LogisticRegression(class_weight="balanced" if args.balanced else None, max_iter=MAX_ITERATIONS, random_state=SEED)
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(train[FEATURES], train["y"])
    s_val = pipeline.predict_proba(val[FEATURES])[:, 1]
    s_test = pipeline.predict_proba(test[FEATURES])[:, 1]
    rows = []
    name = "logreg_balanced" if args.balanced else "logreg"
    evaluate(name, val["y"], s_val, test["y"], s_test, rows)
    res = pd.DataFrame(rows).round(4)#Claude helped me write the results to the terminal and to a csv file.
    print(res.to_string(index=False))
    rkf = KFold(n_splits=5, shuffle=True, random_state=SEED)#I got the initial code from the first assignment but I was getting some errors about inconsistent sample numbers, which I had Claude help me debug.  I also decided to do a random cross-validation test and a time-ordered cross-validation test with some help from Claude.
    tkf = TimeSeriesSplit(n_splits=5)
    train_sorted = train.sort_values("date")
    rfold_scores = cross_val_score(pipeline, train[FEATURES], train_sorted["y"], cv=rkf, scoring="average_precision")
    tfold_scores = cross_val_score(pipeline, train[FEATURES], train_sorted["y"], cv=tkf, scoring="average_precision")
    print("Random Cross-Validation:")
    print(f"fold scores : {np.round(rfold_scores, 4).tolist()}")
    print(f"mean        : {rfold_scores.mean():.4f}")
    print(f"std         : {rfold_scores.std():.4f}")
    print(f"lowest      : {rfold_scores.min():.4f}")
    print(f"highest     : {rfold_scores.max():.4f}")
    print("Time Series Cross-Validation:")
    print(f"fold scores : {np.round(tfold_scores, 4).tolist()}")
    print(f"mean        : {tfold_scores.mean():.4f}")
    print(f"std         : {tfold_scores.std():.4f}")
    print(f"lowest      : {tfold_scores.min():.4f}")
    print(f"highest     : {tfold_scores.max():.4f}")
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    top = np.argsort(np.abs(coefs))[::-1][:10]
    for i in top:
        print(f"{names[i]:<35}{coefs[i]:+.4f}")
    thr = best_f1_threshold(val["y"], s_val)#I had some help from Claude on a good implementation for the confusion matrix.
    y_pred = (s_test >= thr).astype(int)
    disp = ConfusionMatrixDisplay.from_predictions(test["y"], y_pred, display_labels=["On time", "Delayed (15+ min)"], cmap=plt.cm.Blues)
    disp.ax_.set_title(f"{name} (threshold = {thr:.3f})")
    plt.show()
    return
if __name__ == "__main__":
    main()