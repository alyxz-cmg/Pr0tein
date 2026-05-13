"""
Features:
  - length
  - %hydrophobic, %polar, %charged_pos, %charged_neg, %aromatic
  - GRAVY index
  - isoelectric point (pI)
  - net charge at pH 7
  - aromatic fraction (Biopython)
  - instability index
  - 20 amino-acid composition fractions
"""
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from src.config import (
    LABELS_CSV, SEQ_DIR, SEQ_FEATURES_CSV,
    HYDROPHOBIC, POLAR, CHARGED_POS, CHARGED_NEG, AROMATIC,
)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _read_first_chain(fasta_path: Path) -> str:
    """Return the longest chain sequence in a multi-chain RCSB FASTA."""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        return ""
    longest = max(records, key=lambda r: len(r.seq))
    seq = str(longest.seq).replace("X", "").replace("U", "C")
    return "".join(c for c in seq if c in AA_ALPHABET)


def _frac(seq: str, group: set) -> float:
    return sum(1 for a in seq if a in group) / len(seq) if seq else 0.0


def extract(seq: str) -> dict:
    if not seq:
        return {}
    pa = ProteinAnalysis(seq)
    feats = {
        "length": len(seq),
        "frac_hydrophobic": _frac(seq, HYDROPHOBIC),
        "frac_polar": _frac(seq, POLAR),
        "frac_pos": _frac(seq, CHARGED_POS),
        "frac_neg": _frac(seq, CHARGED_NEG),
        "frac_aromatic": _frac(seq, AROMATIC),
        "gravy": pa.gravy(),
        "pI": pa.isoelectric_point(),
        "net_charge_pH7": pa.charge_at_pH(7.0),
        "aromaticity": pa.aromaticity(),
        "instability_index": pa.instability_index(),
    }
    aa_pct = pa.amino_acids_percent
    for aa in AA_ALPHABET:
        feats[f"aa_{aa}"] = aa_pct.get(aa, 0.0)
    return feats


def main():
    df = pd.read_csv(LABELS_CSV)
    rows = []

    for _, row in df.iterrows():
        pdb_id = row["pdb_id"].upper()
        fasta = SEQ_DIR / f"{pdb_id}.fasta"
        seq = _read_first_chain(fasta) if fasta.exists() else ""
        feats = extract(seq)
        feats["pdb_id"] = pdb_id
        feats["sequence"] = seq
        rows.append(feats)

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("No sequence features were generated.")

    cols = ["pdb_id"] + [c for c in out.columns if c not in ("pdb_id", "sequence")] + ["sequence"]
    out = out[cols]

    out.to_csv(SEQ_FEATURES_CSV, index=False)
    print(f"✅ Saved {len(out)} rows × {out.shape[1]} cols → {SEQ_FEATURES_CSV}")

    preview_cols = [c for c in ["pdb_id", "length", "gravy", "pI", "frac_hydrophobic"] if c in out.columns]
    print(out[preview_cols].to_string(index=False))
    
if __name__ == "__main__":
    main()