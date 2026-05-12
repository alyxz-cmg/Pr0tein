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