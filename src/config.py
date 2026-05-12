from pathlib import Path

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SEQ_DIR = RAW_DIR / "sequences"
STRUCT_DIR = RAW_DIR / "structures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset file
LABELS_CSV = DATA_DIR / "labels.csv"

# NCBI Entrez config
ENTREZ_EMAIL = "alexacz1@uci.edu"

# Label map
LABEL_MAP = {0: "single_protofilament", 1: "multi_protofilament"}

# Ensure directories exist
for d in (SEQ_DIR, STRUCT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)