"""
Runs (in order):
  1. fetch_alphafold  (idempotent)
  2. sequence_features
  3. structure_features
  4. merge → data/processed/features.csv

Usage:
    python -m src.build_features
"""
import pandas as pd

from src.config import (
    LABELS_CSV, SEQ_FEATURES_CSV, STRUCT_FEATURES_CSV, FEATURES_CSV,
)
from src import fetch_alphafold, sequence_features, structure_features


def main():
    print("\n[1/4] Fetching AlphaFold monomer models...")
    fetch_alphafold.main()

    from src.config import AF_DIR
    cifs = list(AF_DIR.glob("AF-*.cif"))
    if not cifs:
        print("\n❌ No AlphaFold CIFs were downloaded. Aborting before structure step.")
        print("   Check your internet connection and AlphaFold DB availability,")
        print("   then re-run: python -m src.fetch_alphafold")
        return
    print(f"   Found {len(cifs)} CIF file(s) in {AF_DIR}")

    print("\n[2/4] Extracting sequence features...")
    sequence_features.main()

    print("\n[3/4] Extracting structural features...")
    structure_features.main()

    print("\n[4/4] Merging into final feature matrix...")
    labels = pd.read_csv(LABELS_CSV)
    seq = pd.read_csv(SEQ_FEATURES_CSV)
    struct = pd.read_csv(STRUCT_FEATURES_CSV)

    merged = labels.merge(seq, on="pdb_id", how="left") \
                   .merge(struct, on="pdb_id", how="left")
    merged.to_csv(FEATURES_CSV, index=False)
    print(f"\n✅ Final feature matrix: {merged.shape[0]} rows × {merged.shape[1]} cols")
    print(f"   → {FEATURES_CSV}")
    print("\nLabel balance:")
    print(merged["label"].value_counts().to_string())
    print("\nMissing-value summary (top 5):")
    print(merged.isna().sum().sort_values(ascending=False).head().to_string())


if __name__ == "__main__":
    main()