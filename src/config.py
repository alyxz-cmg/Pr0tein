from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SEQ_DIR = RAW_DIR / "sequences"
STRUCT_DIR = RAW_DIR / "structures"
AF_DIR = DATA_DIR / "alphafold"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

LABELS_CSV = DATA_DIR / "labels.csv"
SEQ_FEATURES_CSV = PROCESSED_DIR / "sequence_features.csv"
STRUCT_FEATURES_CSV = PROCESSED_DIR / "structure_features.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"

ENTREZ_EMAIL = "alexacz1@uci.edu"
LABEL_MAP = {0: "single_protofilament", 1: "multi_protofilament"}

PDB_TO_UNIPROT = {
    # α-synuclein
    "9KAL": "P37840",
    "6XYO": "P37840",
    # Tau
    "7P6A": "P10636",
    "5O3L": "P10636",
    # β-lactoglobulin
    "9IAH": "P02754",
    "9JHF": "P02754",
    # hnRNPA1
    "9GKF": "P09651",
    "7ZJ2": "P09651",
    # Prion protein (PRNP)
    "7LNA": "P04156",
    "7YAT": "P04156",
}

HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQCGP")
CHARGED_POS = set("KRH")
CHARGED_NEG = set("DE")
AROMATIC = set("FWY")

for d in (SEQ_DIR, STRUCT_DIR, AF_DIR, PROCESSED_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOOCV_METRICS_CSV = RESULTS_DIR / "loocv_metrics.csv"
LOOCV_PREDS_CSV = RESULTS_DIR / "loocv_predictions.csv"
TRAINED_MODEL_PKL = RESULTS_DIR / "best_model.pkl"
REPORT_MD = RESULTS_DIR / "REPORT.md"

NON_FEATURE_COLS = [
    "pdb_id", "label", "protein", "variant", "morphology",
    "n_protofilaments", "source", "notes", "sequence",
    "uniprot", "af_available",
]

# Drop due to extraction issues
DROP_FEATURE_COLS = ["plddt_mean", "plddt_median", "frac_disordered"]