import itertools#Claude built this tuning helper function
import time
import pandas as pd
from sklearn.metrics import average_precision_score
def run_tuning(model_name, build_pipeline, param_grid, train, val, features, out_csv):
    """Fit one pipeline per setting on TRAIN, score on VALIDATION, log every trial.
    build_pipeline(**params) must return an unfitted sklearn Pipeline.
    The test month is not an argument, so it cannot influence tuning.
    """
    keys = list(param_grid)
    rows = []
    for i, values in enumerate(itertools.product(*param_grid.values()), start=1):
        params = dict(zip(keys, values))
        t0 = time.time()
        pipe = build_pipeline(**params)
        pipe.fit(train[features], train["y"])
        row = {"model": model_name, "trial": i, **params, "train_pr_auc": average_precision_score(train["y"], pipe.predict_proba(train[features])[:, 1]), "val_pr_auc": average_precision_score(val["y"], pipe.predict_proba(val[features])[:, 1]), "fit_seconds": time.time() - t0}
        rows.append(row)
        print(f"trial {i}: {params} -> val PR-AUC {row['val_pr_auc']:.4f} (train {row['train_pr_auc']:.4f}, {row['fit_seconds']:.0f}s)")
    res = pd.DataFrame(rows)
    res["chosen"] = False
    res.loc[res["val_pr_auc"].idxmax(), "chosen"] = True
    res.round(4).to_csv(out_csv, index=False)
    best = res[res["chosen"]].iloc[0]
    shown = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in keys}
    print(f"\nBest on validation: {shown}  (val PR-AUC {best['val_pr_auc']:.4f}). Saved {out_csv}")
    return res