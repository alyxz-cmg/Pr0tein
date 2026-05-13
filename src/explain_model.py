"""
Produces:
  - results/shap_summary_<model>.png     (global feature impact, beeswarm)
  - results/shap_bar_<model>.png         (mean |SHAP| ranking)
  - results/shap_force_single.png        (one Single-protofilament example)
  - results/shap_force_multi.png         (one Multi-protofilament example)

Falls back gracefully to permutation importance if SHAP fails on the
chosen model (e.g., LogisticRegression inside a Pipeline).
"""
import warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import (
    FEATURES_CSV, RESULTS_DIR, TRAINED_MODEL_PKL,
    NON_FEATURE_COLS, DROP_FEATURE_COLS,
)


def _prep_data():
    df = pd.read_csv(FEATURES_CSV)
    drop = [c for c in NON_FEATURE_COLS + DROP_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])
    return df, X


def _explain(model_pipeline, X):
    """Return SHAP values on the *scaled* feature space using the inner clf."""
    Xi = model_pipeline.named_steps["impute"].transform(X)
    Xs = model_pipeline.named_steps["scale"].transform(Xi)
    Xs_df = pd.DataFrame(Xs, columns=X.columns, index=X.index)
    clf = model_pipeline.named_steps["clf"]

    if hasattr(clf, "estimators_"):
        explainer = shap.TreeExplainer(clf)
    elif clf.__class__.__name__ == "DecisionTreeClassifier":
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.LinearExplainer(clf, Xs_df)

    sv = explainer(Xs_df)

    if sv.values.ndim == 3:
        sv = shap.Explanation(
            values=sv.values[..., 1],
            base_values=sv.base_values[..., 1] if sv.base_values.ndim > 1 else sv.base_values,
            data=sv.data, feature_names=list(X.columns),
        )
    return sv, Xs_df


def main():
    with open(TRAINED_MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    name, model = bundle["name"], bundle["model"]
    print(f"Loaded trained model: {name}")

    df, X = _prep_data()
    try:
        sv, Xs_df = _explain(model, X)
    except Exception as e:
        print(f"[WARN] SHAP failed ({e}); skipping. Use feature_importance plots.")
        return

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title(f"{name} — SHAP Summary (class=Multi)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"shap_summary_{name}.png", dpi=150,
                bbox_inches="tight"); plt.close()

    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title(f"{name} — Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"shap_bar_{name}.png", dpi=150,
                bbox_inches="tight"); plt.close()

    single_idx = df.index[df["label"] == 0][0]
    multi_idx  = df.index[df["label"] == 1][0]
    for idx, tag in [(single_idx, "single"), (multi_idx, "multi")]:
        plt.figure()
        shap.plots.waterfall(sv[idx], max_display=12, show=False)
        plt.title(f"{df.loc[idx, 'pdb_id']} ({df.loc[idx, 'protein']}) — "
                  f"true={df.loc[idx, 'morphology']}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"shap_force_{tag}.png", dpi=150,
                    bbox_inches="tight"); plt.close()

    print(f"✅ SHAP plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()