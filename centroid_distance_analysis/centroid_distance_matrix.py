from time import time
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.topology.tables import vdwradii as vdw_radii
import numpy as np
import sys
from pathlib import Path

# ---- USER SETTINGS ----

INPUT_DIR = Path(sys.argv[1])
NAMED_PDB = Path(sys.argv[2])
GLOB = "*.xtc"
PROBE_RADIUS_A = 1.4  # Angstrem

# -----------------------


def get_vdw_radii(atoms):
    return np.array([vdw_radii.get(atom.element, 1.7) for atom in atoms])


def get_sidechains(universe, nres):
    sidechains = []

    for i in range(len(nres)):
        residue = universe.select_atoms(f"resid {i}")

        sel = residue.atoms.select_atoms(
            "not name N C OA O CB* CG* CT* CBT* CGT* OAT* OT H*"
        )

        if len(sel):
            sidechains.append(
                (
                    residue.resid,
                    sel,
                    get_vdw_radii(sel),
                )
            )

    return sidechains


def compute_overlap_matrix(sidechains):
    n = len(sidechains)

    overlap_mat = np.zeros((n, n))

    for i in range(n):
        overlap_mat[i, i] = 0.0

        _, atoms_i, radii_i = sidechains[i]

        for j in range(i + 1, n):
            _, atoms_j, radii_j = sidechains[j]

            dists = distance_array(
                atoms_i.positions,
                atoms_j.positions,
            )

            # shape: (natoms_i, natoms_j)
            vdw_sum = radii_i[:, np.newaxis] + radii_j[np.newaxis, :]

            overlaps = vdw_sum - dists

            # largest overlap between any atom pair
            max_overlap = overlaps.max()

            overlap_mat[i, j] = max_overlap
            overlap_mat[j, i] = max_overlap

    return overlap_mat


def universe_setup(traj_file, top_file: Path, output_prefix: str = "centroids"):
    """
    Compute sidechain centroid distance matices for all frames.

    Parameters
    traj_file : str, trajectory file name
    top_file : str, topology file name
    output_prefix: str, optional, if given, saves the data in file with a prefix

    Returns
    avg_matrix: np.array
    all_matrices : list of np.ndarray
    resids : np.ndarray
    """

    start_time = time()
    print(f"[START] Loading {traj_file}")

    traj_uni = mda.Universe(top_file, traj_file)
    nres: int = max(r.resid for r in traj_uni.residues) + 1

    n_frames = len(traj_uni.trajectory)
    print(f"Processing {n_frames} frames.")

    all_matrices = []

    for i, ts in enumerate(traj_uni.trajectory):
        resids, sidechains = get_sidechains(traj_uni, nres)

        mat = compute_overlap_matrix(sidechains)

        all_matrices.append(mat)

        print(f"Progress: {100 * i / n_frames:.1f}%", end="\r")

    print("\n")
    all_matrices_array = np.array(all_matrices)
    avg_matrix = np.mean(all_matrices_array, axis=0)

    if output_prefix:
        np.save(f"{output_prefix}_overlap_all_matrices.npy", all_matrices_array)
        np.savetxt(f"{output_prefix}_overlap_resids.dat", resids, fmt="%.3f")

    print(f"[DONE] {traj_file} in {time() - start_time:.2f}\n")

    return avg_matrix, all_matrices, resids


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print(f"No files found in {INPUT_DIR} matching {GLOB}.")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")

    avg_matrix, all_matrices, resids = universe_setup(files, NAMED_PDB)
