"""
Pulls metrics + figures produced by train_model.py and explain_model.py
into a single REPORT.md ready for the final write-up.
"""
import pandas as pd
from datetime import datetime

from src.config import (
    RESULTS_DIR, LOOCV_METRICS_CSV, LOOCV_PREDS_CSV, REPORT_MD,
    FEATURES_CSV,
)


def _table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def main():
    metrics = pd.read_csv(LOOCV_METRICS_CSV)
    preds = pd.read_csv(LOOCV_PREDS_CSV)
    feats = pd.read_csv(FEATURES_CSV)

    best = metrics.iloc[0]
    by_model = preds.groupby("model")["correct"].mean().round(3)
    misses = preds[preds["correct"] == 0].sort_values(["model", "pdb_id"])

    md = f"""# Amyloid Fibril Morphology Predictor — Results Report

*Auto-generated {datetime.now().strftime("%Y-%m-%d %H:%M")}*

## 1. Dataset

- **N samples:** {len(feats)}
- **Class balance:** {dict(feats['label'].value_counts().sort_index())}
- **Feature matrix:** `data/processed/features.csv` ({feats.shape[1]} columns)
- **Dropped due to Step-2 extraction limits:** `plddt_mean`, `plddt_median`,
`frac_disordered` (AF v6 mmCIF format change — pLDDT lives in a separate
metric block).

## 2. Validation Scheme

- **Leave-One-Out Cross-Validation (LOOCV)** — 10 folds, deterministic, no
random seed bias. Selected because n=10 makes k-fold splits highly
sensitive to test-set composition (the deep-research report flags
*"data scarcity"* and *"overfitting"* as primary risks; LOOCV is the
canonical mitigation for both).

## 3. LOOCV Metrics

{_table(metrics)}

**Best model: `{best['model']}`** (Accuracy = {best['accuracy']:.3f},
F1 = {best['f1']:.3f}, ROC-AUC = {best['roc_auc']:.3f}).

### Per-model accuracy
{_table(by_model.reset_index().rename(columns={'correct': 'accuracy'}))}

## 4. Misclassified Examples (Student A — biological review)

{_table(misses) if len(misses) else "_None — all 10 samples classified correctly._"}

> Action for **Student A**: examine the structural details of the misses
> above. Are they "twisted vs. flat" doublets? Brain-derived vs. in vitro?
> These outliers are the most informative entries for biological discussion.

## 5. Figures

### Confusion matrices
![LR](confusion_matrix_LogisticRegression.png)
![DT](confusion_matrix_DecisionTree.png)
![RF](confusion_matrix_RandomForest.png)

### Feature importance / coefficients
![LR](feature_importance_LogisticRegression.png)
![DT](feature_importance_DecisionTree.png)
![RF](feature_importance_RandomForest.png)

### SHAP — global
![Beeswarm](shap_summary_{best['model']}.png)
![Bar](shap_bar_{best['model']}.png)

### SHAP — per-sample (Student A interpretation targets)
![Single](shap_force_single.png)
![Multi](shap_force_multi.png)

## 6. Interpretation Notes (template for Student A)

- The **top-ranked features** in §5 are the biological drivers of fibril
morphology in this dataset. Cross-reference each against the deep-research
report (§5 Feature Sets) — features like `frac_hydrophobic`, `gravy`,
`ss_sheet`, and `sasa_hydrophobic_frac` map directly to the steric-zipper
/ lateral-association literature (Iadanza et al., 2018).
- The **SHAP force plots** show *why* a specific protein was classified
Single vs. Multi. Use these in the report's Discussion section to argue
beyond accuracy numbers.
- All structural features are **identical within UniProt groups** (the four
APP entries share Rg, SS%, SASA). Discriminative power therefore comes
from sequence features (chain-specific FASTA) interacting with structural
context — a known limitation noted in §7 (Risks & Mitigations) of the
deep-research report.

## 7. Reproducibility

```bash
pip install -r requirements.txt
python -m src.fetch_data           # Step 1
python -m src.build_features       # Step 2
python -m src.train_model          # Step 3a (this report)
python -m src.explain_model        # Step 3b
python -m src.generate_report      # Step 3c

"""

    REPORT_MD.write_text(md)
    print(f"✅ Report written: {REPORT_MD}")

if __name__ == "__main__":
    main()