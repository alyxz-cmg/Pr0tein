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
    "7Q4B": "P05067",
    "2NAO": "P05067",
    "6W06": "P05067",
    "2M4J": "P05067",
    "8RRR": "P37840",
    "6XYO": "P37840",
    "7P6A": "P10636",
    "5O3L": "P10636",
    "7P0V": "P61769",
    "6GK3": "P61769",
}

HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQCGP")
CHARGED_POS = set("KRH")
CHARGED_NEG = set("DE")
AROMATIC = set("FWY")

for d in (SEQ_DIR, STRUCT_DIR, AF_DIR, PROCESSED_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)