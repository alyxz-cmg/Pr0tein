"""
For each PDB entry (mapped via UniProt → AF model), computes:
  - radius of gyration
  - mean/median pLDDT (AF confidence as a proxy for monomer disorder)
  - fraction of residues with pLDDT < 50 (predicted disorder)
  - secondary structure %: helix, sheet, coil  (biotite annotate_sse)
  - total SASA, polar SASA, hydrophobic SASA  (freesasa)
  - relative hydrophobic SASA (hydrophobic / total)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import biotite.structure.io.pdbx as pdbx
import biotite.structure as struc
import freesasa

from src.config import (
    LABELS_CSV, AF_DIR, STRUCT_FEATURES_CSV, PDB_TO_UNIPROT, HYDROPHOBIC,
)

THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


def _load_structure(cif_path: Path):
    cif_file = pdbx.CIFFile.read(cif_path)
    atoms = pdbx.get_structure(cif_file, model=1)
    atoms = atoms[atoms.hetero == False]
    return atoms


def _radius_of_gyration(atoms) -> float:
    coords = atoms.coord
    masses = np.array([struc.info.mass(a) for a in atoms.element])
    com = np.average(coords, axis=0, weights=masses)
    rg2 = np.average(np.sum((coords - com) ** 2, axis=1), weights=masses)
    return float(np.sqrt(rg2))


def _plddt_stats(atoms) -> dict:
    # In AF mmCIFs, pLDDT is stored in the B-factor field (per-atom).
    ca_mask = atoms.atom_name == "CA"
    plddt = atoms.b_factor[ca_mask]
    return {
        "plddt_mean": float(np.mean(plddt)),
        "plddt_median": float(np.median(plddt)),
        "frac_disordered": float(np.mean(plddt < 50.0)),
    }


def _ss_fractions(atoms) -> dict:
    # Biotite returns 'a' (alpha-helix), 'b' (beta-sheet), 'c' (coil)
    ca = atoms[atoms.atom_name == "CA"]
    try:
        sse = struc.annotate_sse(atoms)
    except Exception:
        sse = np.array(["c"] * len(ca))
    n = max(len(sse), 1)
    return {
        "ss_helix": float(np.sum(sse == "a") / n),
        "ss_sheet": float(np.sum(sse == "b") / n),
        "ss_coil":  float(np.sum(sse == "c") / n),
    }


def _sasa_features(cif_path: Path, atoms) -> dict:
    # freesasa needs a PDB-style file; write a temp PDB from biotite atoms.
    import tempfile, biotite.structure.io.pdb as pdb_io
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        pdb_file = pdb_io.PDBFile()
        pdb_file.set_structure(atoms)
        pdb_file.write(f.name)
        tmp = f.name

    structure = freesasa.Structure(tmp)
    result = freesasa.Calc().calculate(structure)
    total = result.totalArea()

    # Per-residue SASA → split by hydrophobicity of residue type
    residues = result.residueAreas()
    hydro, polar = 0.0, 0.0
    for chain in residues.values():
        for res in chain.values():
            one = THREE_TO_ONE.get(res.residueType, "X")
            if one in HYDROPHOBIC:
                hydro += res.total
            else:
                polar += res.total

    return {
        "sasa_total": float(total),
        "sasa_hydrophobic": float(hydro),
        "sasa_polar": float(polar),
        "sasa_hydrophobic_frac": float(hydro / total) if total else 0.0,
    }


def extract_for_pdb(pdb_id: str) -> dict:
    uniprot = PDB_TO_UNIPROT[pdb_id]
    cif = AF_DIR / f"AF-{uniprot}-F1.cif"
    if not cif.exists():
        return {"pdb_id": pdb_id, "uniprot": uniprot, "af_available": False}

    atoms = _load_structure(cif)
    feats = {"pdb_id": pdb_id, "uniprot": uniprot, "af_available": True,
             "n_residues": int(len(set(zip(atoms.chain_id, atoms.res_id))))}
    feats["radius_of_gyration"] = _radius_of_gyration(atoms)
    feats.update(_plddt_stats(atoms))
    feats.update(_ss_fractions(atoms))
    feats.update(_sasa_features(cif, atoms))
    return feats


def main():
    df = pd.read_csv(LABELS_CSV)
    rows = []
    for pdb_id in tqdm(df["pdb_id"].str.upper(), desc="Structural features"):
        try:
            rows.append(extract_for_pdb(pdb_id))
        except Exception as e:
            print(f"[ERROR] {pdb_id}: {e}")
            rows.append({"pdb_id": pdb_id, "af_available": False})

    out = pd.DataFrame(rows)
    out.to_csv(STRUCT_FEATURES_CSV, index=False)
    print(f"✅ Saved {len(out)} rows × {out.shape[1]} cols → {STRUCT_FEATURES_CSV}")
    print(out[["pdb_id", "radius_of_gyration", "plddt_mean",
               "ss_sheet", "sasa_hydrophobic_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()