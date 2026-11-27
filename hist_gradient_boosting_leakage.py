from helpers import chrono_split, evaluate, load, best_f1_threshold, SEED, MAX_ITERATIONS, LEAKAGE_CATEGORICAL, LEAKAGE_NUMERICAL, LEAKAGE_FEATURES
import argparse
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
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
    rkf = KFold(n_splits=5, shuffle=True, random_state=SEED)#I got the initial code from the first assignment but I was getting some errors about inconsistent sample numbers, which I had Claude help me debug.  I also decided to do a random cross-validation test and a time-ordered cross-validation test with some help from Claude.
    tkf = TimeSeriesSplit(n_splits=5)
    train_sorted = train.sort_values("date")
    rfold_scores = cross_val_score(pipeline, train_sorted[LEAKAGE_FEATURES], train_sorted["y"], cv=rkf, scoring="average_precision")
    tfold_scores = cross_val_score(pipeline, train_sorted[LEAKAGE_FEATURES], train_sorted["y"], cv=tkf, scoring="average_precision")
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
    thr = best_f1_threshold(val["y"], s_val)
    y_pred = (s_test >= thr).astype(int)
    disp = ConfusionMatrixDisplay.from_predictions(test["y"], y_pred, display_labels=["On time", "Delayed (15+ min)"], cmap=plt.cm.Blues)
    disp.ax_.set_title(f"{name} (threshold = {thr:.3f})")
    plt.show()
    return
if __name__ == "__main__":
    main()