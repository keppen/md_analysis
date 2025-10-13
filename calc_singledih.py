from pathlib import Path
import sys
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral


def compute_torsion_dataframe(ref_universe, nres, atom_selector_fn, verbose=True):
    """
    Select atoms for each residue using `atom_selector_fn` and compute dihedral angles.
    Returns a DataFrame of shape (n_frames, n_residues).
    """
    dihedral_groups = []
    for i in range(nres):
        sel = atom_selector_fn(ref_universe, i, nres)
        if not all(len(group) == 1 for group in sel):
            print(f"Residue {i} skipped due to missing atoms: {[len(g) for g in sel]}")
            continue
        dihedral_groups.append(sel[0] + sel[1] + sel[2] + sel[3])

    if not dihedral_groups:
        print("No valid dihedral groups found.")
        return pd.DataFrame()

    dih = Dihedral(dihedral_groups)
    dih.run(verbose=verbose, stop=1)

    angles = dih.results.angles  # shape (n_frames, n_dihedrals)

    # Return DataFrame with 1 column per dihedral group
    return pd.DataFrame(angles)


# Custom torsion definitions for your polymer
def selector_torsion1(uni, i, nres):
    """PHI angle"""
    if i == 0:
        return [
            uni.select_atoms("resid 0 and name CT"),
            uni.select_atoms("resid 0 and name N"),
            uni.select_atoms("resid 0 and name CG"),
            uni.select_atoms("resid 0 and name CB"),
        ]
    else:
        return [
            uni.select_atoms(f"resid {i - 1} and name C"),
            uni.select_atoms(f"resid {i} and name N"),
            uni.select_atoms(f"resid {i} and name CG"),
            uni.select_atoms(f"resid {i} and name CB"),
        ]


def selector_torsion2(uni, i, nres):
    """XI angle"""
    return [
        uni.select_atoms(f"resid {i} and name N"),
        uni.select_atoms(f"resid {i} and name CG"),
        uni.select_atoms(f"resid {i} and name CB"),
        uni.select_atoms(f"resid {i} and name OA"),
    ]


def selector_torsion3(uni, i, nres):
    """CHI angle"""
    if i == nres - 1:
        next_atom = uni.select_atoms(f"resid {i} and name HO")
    else:
        next_atom = uni.select_atoms(f"resid {i} and name C")
    return [
        uni.select_atoms(f"resid {i} and name CG"),
        uni.select_atoms(f"resid {i} and name CB"),
        uni.select_atoms(f"resid {i} and name OA"),
        next_atom,
    ]


if __name__ == "__main__":
    ref_file = Path(sys.argv[1])
    traj_file = Path(sys.argv[2])

    # Load reference and trajectory
    ref = mda.Universe(ref_file)
    traj = mda.Universe(traj_file)

    if len(ref.atoms) != len(traj.atoms):
        raise ValueError("Atom counts don't match")

    ref.trajectory = traj.trajectory
    nres = max(r.resid for r in ref.residues) + 1

    # Compute torsions
    torsion1 = compute_torsion_dataframe(ref, nres, selector_torsion1, verbose=False)
    torsion2 = compute_torsion_dataframe(ref, nres, selector_torsion2, verbose=False)
    torsion3 = compute_torsion_dataframe(ref, nres, selector_torsion3, verbose=False)

    # Rename columns for clarity
    # torsion1.columns = [f"phi_{i}" for i in range(torsion1.shape[1])]
    # torsion2.columns = [f"xi_{i}" for i in range(torsion2.shape[1])]
    # torsion3.columns = [f"chi_{i}" for i in range(torsion3.shape[1])]

    # Each torsion is shape (n_frames, n_residues); we assume 1 frame, so transpose to (n_residues, 1)
    df = pd.concat([torsion1.T, torsion2.T, torsion3.T], axis=1)
    df.columns = ["φ", "ξ", "χ"]
    df.index = [f"{i + 1}" for i in range(df.shape[0])]
    df.index.name = "mer"  # 👈 Add name to index column
    df.reset_index(inplace=True)
    # df.to_csv("torsions.csv", index=False)
    # print(df.to_string(index=False, float_format=lambda x: f"{x:.0f}"))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.0f}"))

    # df.to_csv("torsions.csv", index=False)
    # print(df)
