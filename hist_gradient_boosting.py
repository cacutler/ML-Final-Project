from helpers import chrono_split, evaluate, load, best_f1_threshold, SEED, MAX_ITERATIONS, CATEGORICAL, NUMERICAL, FEATURES
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
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score
def build_pipeline(balanced=False, learning_rate=0.1, max_iterations=MAX_ITERATIONS, max_leaf_nodes=31):
    preprocessor = ColumnTransformer(transformers=[("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CATEGORICAL), ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERICAL)])#Claude helped me rework the preprocessor pipeline for the hist gradient boosting model.
    model = HistGradientBoostingClassifier(categorical_features=list(range(len(CATEGORICAL))), class_weight="balanced" if balanced else None, early_stopping=False, learning_rate=learning_rate, max_iter=max_iterations, max_leaf_nodes=max_leaf_nodes, random_state=SEED)#Claude helped me in pointing out the parameters I should use for the hist gradient boosting classifier.
    return Pipeline([("prep", preprocessor), ("model", model)])
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./")
    ap.add_argument("--airports", nargs="+", required=True)
    ap.add_argument("--balanced", action="store_true", help="use class_weight='balanced' (default: no class weights)")
    ap.add_argument("--tune", action="store_true", help="run the hyperparameter search on train/validation")
    ap.add_argument("--out", default="gradient_boosting_results.csv")
    args = ap.parse_args()
    df = load(args.data_dir, args.airports)
    train, val, test = chrono_split(df)
    if args.tune:
        rows = []
        for learning_rate in [0.05, 0.1]:
            for max_iterations in [100, 200, 400]:
                for max_leaf_nodes in [15, 31]:
                    pipeline = build_pipeline(balanced=args.balanced, learning_rate=learning_rate, max_iterations=max_iterations, max_leaf_nodes=max_leaf_nodes)
                    pipeline.fit(train[FEATURES], train["y"])
                    rows.append({"learning_rate": learning_rate, "max_iterations": max_iterations, "max_leaf_nodes": max_leaf_nodes, "train_pr_auc": average_precision_score(train["y"], pipeline.predict_proba(train[FEATURES])[:, 1]), "val_pr_auc": average_precision_score(val["y"], pipeline.predict_proba(val[FEATURES])[:, 1])})
        log = pd.DataFrame(rows).round(4)
        best = log.iloc[int(log["val_pr_auc"].argmax())]
        log["chosen"] = np.arange(len(log)) == int(log["val_pr_auc"].argmax())
        settings = {"learning_rate": float(best["learning_rate"]), "max_iterations": int(best["max_iterations"]), "max_leaf_nodes": int(best["max_leaf_nodes"])}
    else:
        settings = {"learning_rate": 0.1, "max_iterations": 200, "max_leaf_nodes": 31}
    pipeline = build_pipeline(balanced=args.balanced, **settings)
    pipeline.fit(train[FEATURES], train["y"])
    s_val = pipeline.predict_proba(val[FEATURES])[:, 1]
    s_test = pipeline.predict_proba(test[FEATURES])[:, 1]
    rows = []
    name = "histgradientboosting_balanced" if args.balanced else "histgradientboosting"
    evaluate(name, val["y"], s_val, test["y"], s_test, rows)
    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False))
    rkf = KFold(n_splits=5, shuffle=True, random_state=SEED)#I got the initial code from the first assignment but I was getting some errors about inconsistent sample numbers, which I had Claude help me debug.  I also decided to do a random cross-validation test and a time-ordered cross-validation test with some help from Claude.
    tkf = TimeSeriesSplit(n_splits=5)
    train_sorted = train.sort_values("date")#Claude fixed a bug with the sorted training data
    rfold_scores = cross_val_score(pipeline, train_sorted[FEATURES], train_sorted["y"], cv=rkf, scoring="average_precision")
    tfold_scores = cross_val_score(pipeline, train_sorted[FEATURES], train_sorted["y"], cv=tkf, scoring="average_precision")
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