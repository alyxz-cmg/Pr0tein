"""
Usage:
    python -m src.fetch_alphafold
"""
import sys
import time
import requests
import pandas as pd
from tqdm import tqdm

from src.config import LABELS_CSV, AF_DIR, PDB_TO_UNIPROT

AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.cif"


def fetch_af_model(uniprot: str) -> bool:
    out = AF_DIR / f"AF-{uniprot}-F1.cif"
    if out.exists():
        return True
    try:
        r = requests.get(AF_URL.format(uniprot=uniprot), timeout=60)
        if r.ok and "data_" in r.text[:200]:
            out.write_text(r.text)
            return True
        print(f"[WARN] AF model not available: {uniprot} ({r.status_code})",
              file=sys.stderr)
    except requests.RequestException as e:
        print(f"[WARN] {uniprot}: {e}", file=sys.stderr)
    return False


def main():
    df = pd.read_csv(LABELS_CSV)
    uniprots = sorted({PDB_TO_UNIPROT[p.upper()] for p in df["pdb_id"]})
    print(f"Fetching {len(uniprots)} unique AlphaFold monomer models...")

    rows = []
    for u in tqdm(uniprots, desc="AlphaFold DB"):
        ok = fetch_af_model(u)
        rows.append({"uniprot": u, "ok": ok})
        time.sleep(0.3)

    print("\n=== AlphaFold Fetch Report ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved to: {AF_DIR}")


if __name__ == "__main__":
    main()