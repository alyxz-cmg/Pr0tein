"""
Trains three interpretable classifiers (Logistic Regression, Decision Tree,
Random Forest), evaluates each with LOOCV (n=10 folds, deterministic),
and writes per-model metrics + per-sample predictions to results/.
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)

from src.config import (
    FEATURES_CSV, RESULTS_DIR, LOOCV_METRICS_CSV, LOOCV_PREDS_CSV,
    TRAINED_MODEL_PKL, NON_FEATURE_COLS, DROP_FEATURE_COLS,
)

RANDOM_STATE = 42


def load_features():
    """Load features.csv and split into (X, y, pdb_ids, feature_names)."""
    df = pd.read_csv(FEATURES_CSV)
    drop_cols = [c for c in NON_FEATURE_COLS + DROP_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df["label"].astype(int).values
    pdb_ids = df["pdb_id"].values
    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Dropped columns: {drop_cols}")
    return X, y, pdb_ids, list(X.columns)


def make_models():
    """Return three pipelines (impute → scale → classify)."""
    base = lambda clf: Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    clf),
    ])
    return {
        "LogisticRegression": base(LogisticRegression(
            max_iter=5000, C=1.0, random_state=RANDOM_STATE)),
        "DecisionTree": base(DecisionTreeClassifier(
            max_depth=4, random_state=RANDOM_STATE)),
        "RandomForest": base(RandomForestClassifier(
            n_estimators=200, max_depth=4, random_state=RANDOM_STATE)),
    }


def loocv_evaluate(model, X, y, pdb_ids, model_name):
    """Run LOOCV; return preds, probas, fold-by-fold records."""
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)
    y_proba = np.zeros(len(y), dtype=float)
    records = []

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        pred = int(model.predict(X_te)[0])
        proba = float(model.predict_proba(X_te)[0, 1])
        y_pred[test_idx[0]] = pred
        y_proba[test_idx[0]] = proba
        records.append({
            "model": model_name,
            "pdb_id": pdb_ids[test_idx[0]],
            "y_true": int(y_te[0]),
            "y_pred": pred,
            "proba_multi": proba,
            "correct": int(pred == y_te[0]),
        })
    return y_pred, y_proba, records


def plot_confusion(y_true, y_pred, name, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Single", "Multi"],
                yticklabels=["Single", "Multi"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} — LOOCV Confusion Matrix")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_feature_importance(model, feature_names, name, out_path, top_k=15):
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        title = f"{name} — Feature Importance"
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
        title = f"{name} — |Coefficient|"
    else:
        return
    s = pd.Series(imp, index=feature_names).sort_values(ascending=True).tail(top_k)
    fig, ax = plt.subplots(figsize=(6, 5))
    s.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(title); ax.set_xlabel("Importance")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def main():
    X, y, pdb_ids, feature_names = load_features()
    models = make_models()

    metrics_rows, all_preds = [], []

    for name, model in models.items():
        print(f"\n--- {name} ---")
        y_pred, y_proba, records = loocv_evaluate(model, X, y, pdb_ids, name)
        all_preds.extend(records)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y, y_proba)
        except ValueError:
            auc = float("nan")
        print(f"  Accuracy: {acc:.3f}   F1: {f1:.3f}   ROC-AUC: {auc:.3f}")
        metrics_rows.append({
            "model": name, "accuracy": acc, "f1": f1, "roc_auc": auc,
        })

        model.fit(X, y)
        plot_confusion(y, y_pred, name,
                       RESULTS_DIR / f"confusion_matrix_{name}.png")
        plot_feature_importance(model, feature_names, name,
                                RESULTS_DIR / f"feature_importance_{name}.png")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("accuracy", ascending=False)
    metrics_df.to_csv(LOOCV_METRICS_CSV, index=False)
    pd.DataFrame(all_preds).to_csv(LOOCV_PREDS_CSV, index=False)

    print("\n=== LOOCV Summary ===")
    print(metrics_df.to_string(index=False))

    best_name = metrics_df.iloc[0]["model"]
    best = make_models()[best_name].fit(X, y)
    with open(TRAINED_MODEL_PKL, "wb") as f:
        pickle.dump({"name": best_name, "model": best,
                     "features": feature_names}, f)
    print(f"\n💾 Best model: {best_name} → {TRAINED_MODEL_PKL.name}")


if __name__ == "__main__":
    main()