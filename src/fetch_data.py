"""
Usage:
    python -m src.fetch_data
"""
import sys
import time
import requests
import pandas as pd
from tqdm import tqdm
from Bio import Entrez, SeqIO

from src.config import LABELS_CSV, SEQ_DIR, STRUCT_DIR, ENTREZ_EMAIL

Entrez.email = ENTREZ_EMAIL
RCSB_FASTA = "https://www.rcsb.org/fasta/entry/{pdb_id}"
RCSB_PDB = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_fasta(pdb_id: str) -> str | None:
    """Fetch FASTA for a PDB ID from RCSB (primary) with NCBI fallback."""
    out_path = SEQ_DIR / f"{pdb_id}.fasta"
    if out_path.exists():
        return out_path.read_text()

    # Primary: RCSB
    try:
        r = requests.get(RCSB_FASTA.format(pdb_id=pdb_id), timeout=30)
        if r.ok and r.text.startswith(">"):
            out_path.write_text(r.text)
            return r.text
    except requests.RequestException:
        pass

    # Fallback: NCBI Entrez (Protein db)
    try:
        handle = Entrez.esearch(db="protein", term=f"{pdb_id}[PDB]")
        ids = Entrez.read(handle)["IdList"]
        handle.close()
        if ids:
            handle = Entrez.efetch(
                db="protein", id=ids[0], rettype="fasta", retmode="text"
            )
            txt = handle.read()
            handle.close()
            out_path.write_text(txt)
            return txt
    except Exception as e:
        print(f"[WARN] NCBI fallback failed for {pdb_id}: {e}", file=sys.stderr)

    return None


def fetch_pdb(pdb_id: str) -> bool:
    """Download PDB coordinate file from RCSB."""
    out_path = STRUCT_DIR / f"{pdb_id}.pdb"
    if out_path.exists():
        return True
    try:
        r = requests.get(RCSB_PDB.format(pdb_id=pdb_id), timeout=60)
        if r.ok and r.text.startswith(("HEADER", "ATOM", "REMARK")):
            out_path.write_text(r.text)
            return True
    except requests.RequestException as e:
        print(f"[WARN] PDB fetch failed for {pdb_id}: {e}", file=sys.stderr)
    return False


def main():
    df = pd.read_csv(LABELS_CSV)
    print(f"Loaded {len(df)} entries from {LABELS_CSV.name}")
    print(f"Label balance:\n{df['label'].value_counts().to_string()}\n")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching"):
        pdb_id = row["pdb_id"].strip().upper()
        fasta = fetch_fasta(pdb_id)
        pdb_ok = fetch_pdb(pdb_id)
        results.append({
            "pdb_id": pdb_id,
            "fasta_ok": fasta is not None,
            "pdb_ok": pdb_ok,
        })
        time.sleep(0.34)

    report = pd.DataFrame(results)
    print("\n=== Fetch Report ===")
    print(report.to_string(index=False))
    print(f"\nFASTA saved to: {SEQ_DIR}")
    print(f"PDB saved to:   {STRUCT_DIR}")


if __name__ == "__main__":
    main()