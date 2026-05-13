# Amyloid Fibril Morphology Predictor

An interpretable ML pipeline that predicts amyloid fibril **morphology class**
(single- vs. multi-protofilament) from monomer sequence and predicted
3D structure. Built for CS 189 project @ UC Irvine.

## 🎯 Project Goal

Amyloid fibrils are highly polymorphic: the same protein sequence can form
multiple distinct folds. This project trains a small, explainable classifier
on curated cryo-EM / NMR fibril structures to predict whether a given
amyloidogenic sequence will assemble as a **single protofilament** or
**multi-protofilament (doublet/triplet)** fibril.

## 🧪 Binary Labeling Scheme

| Label | Class                          | Description |
|-------|--------------------------------|-------------|
| `0`   | Single protofilament           | One continuous cross-β chain |
| `1`   | Multi-protofilament            | Two or more laterally associated protofilaments (doublets, triplets) |

**Why binary?**
- **Biological relevance:** Lateral association involves distinct "steric
  zipper" interfaces (e.g., NAC region in α-synuclein) that are absent in
  single-protofilament forms — clear patterns for the model to learn.
- **Data availability:** Recent cryo-EM advances have populated the PDB with
  both classes at high resolution.
- **Computational feasibility:** Predicting the exact protofilament count
  (2 / 3 / 4) is a harder multiclass/regression problem that small datasets
  cannot support reliably.

## 📊 Core Dataset

10 PDB entries, balanced 5 single / 5 multi across 5 amyloidogenic proteins:

| Protein | Single (label=0) | Multi (label=1) |
|---------|------------------|-----------------|
| Amyloid-β (1-42) | **7Q4B** (brain Type I) | **2NAO** (double-horseshoe doublet) |
| Amyloid-β (1-40) | **6W06** (in vitro) | **2M4J** (3-fold triplet) |
| α-Synuclein | **8RRR** (Lewy fold) | **6XYO** (MSA Type I doublet) |
| Tau | **7P6A** (Pick's disease) | **5O3L** (PHF doublet) |
| β2-Microglobulin | **7P0V** | **6GK3** (doublet) |

Full metadata is in [`data/labels.csv`](data/labels.csv).

## 📁 Repository Structure

```
amyloid-morphology-predictor/
├── data/
│   ├── labels.csv          # Core dataset (PDB IDs + binary labels)
│   └── raw/
│       ├── sequences/      # FASTA files (fetched)
│       └── structures/     # PDB files (fetched)
├── src/
│   ├── config.py           # Paths, constants
│   └── fetch_data.py       # NCBI / RCSB fetcher
├── notebooks/              # Exploration & analysis
├── results/                # Figures, metrics, trained models
├── requirements.txt
└── README.md
```

## 🚀 Setup & Quickstart

```bash
# 1. Clone & enter the repo
git clone <your-repo-url> amyloid-morphology-predictor
cd amyloid-morphology-predictor

# 2. Create a virtual environment (Python 3.10 recommended)
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your NCBI email in src/config.py  (required by Entrez policy)
#    ENTREZ_EMAIL = "you@uci.edu"

# 5. Fetch FASTA and PDB files for the core dataset
python -m src.fetch_data
```

After the fetch completes, you should see FASTA files in
`data/raw/sequences/` and PDB files in `data/raw/structures/`.

## Feature Engineering

We extract three feature families per protein and merge them into a single
matrix at `data/processed/features.csv`.

### Why AlphaFold-predicted monomers?
Following the recommendation in our research notes, we include monomer
structures from the **AlphaFold Protein Structure Database (EBI)** to enable
biologically meaningful descriptors (SASA, Rg, secondary-structure content)
that pure sequence features cannot capture. PDB IDs `9LIW` and `9D23` were
**excluded** because they are not yet publicly released and break the
automated pipeline; the 10 confirmed IDs from Step 1 remain the core dataset.

### Feature families

| Family | Examples | Source |
|--------|----------|--------|
| **Sequence** | length, AA composition (20-d), %hydrophobic/polar/charged/aromatic | `src/sequence_features.py` |
| **Physicochemical** | GRAVY, pI, net charge @ pH7, aromaticity, instability index | Biopython `ProteinAnalysis` |
| **Structural (AF monomer)** | Rg, mean/median pLDDT, %disordered, helix/sheet/coil %, total/polar/hydrophobic SASA | `src/structure_features.py` |

### Run it
```bash
python -m src.build_features
```
Outputs:
- `data/processed/sequence_features.csv`
- `data/processed/structure_features.csv`
- `data/processed/features.csv`  ← used in Step 3

### Notes
- **pLDDT as disorder proxy:** AF stores per-atom pLDDT in the B-factor
  column. Low pLDDT in monomeric amyloidogenic proteins (Aβ, α-syn, tau) is
  expected and biologically informative.
- **UniProt mapping:** Each PDB → parent UniProt (`PDB_TO_UNIPROT` in
  `config.py`). Multiple PDBs share a UniProt (e.g., 4× APP entries) — they
  receive identical structural features but distinct sequence/label rows.
- **No external DSSP binary** required: we use Biotite's `annotate_sse`
  for SS and `freesasa` (pure Python) for SASA.

## Model Training, Explainability & Reporting

### Run the full intelligence layer
```bash
python -m src.train_model        # LOOCV training + metrics + figures
python -m src.explain_model      # SHAP summary + force plots
python -m src.generate_report    # Auto-builds results/REPORT.md
```

### What's produced
| File | Contents |
|---|---|
| `results/loocv_metrics.csv` | Accuracy / F1 / ROC-AUC for LR, DT, RF |
| `results/loocv_predictions.csv` | Per-PDB y_true, y_pred, proba_multi, correct |
| `results/confusion_matrix_*.png` | Per-model 2×2 confusion |
| `results/feature_importance_*.png` | Coefficient or impurity importance |
| `results/shap_summary_*.png` | Global SHAP beeswarm (best model) |
| `results/shap_force_{single,multi}.png` | Per-protein waterfall plots |
| `results/best_model.pkl` | Pickled pipeline (impute → scale → clf) |
| `results/REPORT.md` | Auto-generated final report |

### Validation choice — LOOCV
With n=10, a 5-fold split would put 2 proteins in each test fold, making
metrics extremely sensitive to the random shuffle. LOOCV is deterministic
(no seed bias), maximizes training data per fold (9/10), and is the
canonical small-sample CV for structural biology. This directly addresses
*"Overfitting: small sample size may cause overfitting → use simple models,
cross-validation, and limit feature dimensionality"* in §7 of the
deep-research report.

### Explainability — coefficients + SHAP
- **scikit-learn coefficients / feature importances** answer *"which
  features matter on average?"* (the **Student B** lens).
- **SHAP** answers *"why was THIS specific protein classified this way?"*
  — exactly the per-sample insight **Student A** needs to discuss
  biological outliers (e.g., a Tau PHF that the model misclassifies because
  its monomer hydrophobicity profile resembles a single-protofilament
  fold).

### Known limitations (carried into the report)
- pLDDT-derived features (`plddt_mean`, `plddt_median`, `frac_disordered`)
  are dropped — the AlphaFold v6 mmCIF format moved pLDDT out of the
  B-factor field into a `_ma_qa_metric_local` block. Not a Step-3 blocker;
  Rg, SS%, and SASA remain.
- Same-UniProt rows share structural features (4× APP, 2× α-syn, 2× tau,
  2× β2m). Discrimination within a UniProt group depends on sequence
  features. The classifier still has to learn a consistent decision
  boundary across all four proteins, which is the genuine ML challenge.

## 👥 Team Roles

| Student | Focus | Step 1 Responsibilities |
|---------|-------|-------------------------|
| **Rayhan** | Biology / literature | Curated PDB list, verified morphology labels, wrote biological rationale in this README |
| **Alexander** | CS / ML | Environment setup, `fetch_data.py`, project scaffolding, dependency pinning |

## 📚 Data Sources

- **RCSB PDB** — structure coordinates (`https://files.rcsb.org/download/<ID>.pdb`)
- **NCBI MMDB / Protein** — FASTA sequences via Biopython `Entrez.efetch`
- **AlphaFold DB** (Step 2) — predicted monomer structures

All data are public-domain / open-access; no licensing restrictions.

## 📜 License

Code: MIT. Data: public domain (RCSB / NCBI).