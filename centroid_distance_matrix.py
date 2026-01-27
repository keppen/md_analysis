from time import time
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
import sys
from pathlib import Path

# ---- USER SETTINGS ----

INPUT_DIR = Path(sys.argv[1])
NAMED_PDB = Path(sys.argv[2])
GLOB = "*.xtc"
PROBE_RADIUS_A = 1.4  # Angstrem

# -----------------------

PROBE_RADIUS_NM = PROBE_RADIUS_A / 10  # nm


def compute_sidechain_centrorids(universe: mda.Universe, nres: int):
    centroids = []
    resids = []

    for i in range(nres):
        selection = f"resid {i}"
        residue = universe.select_atoms(selection)
        sidechain = residue.atoms.select_atoms(
            "not name N C OA O CB* CG* CT* CBT* CGT* OAT* OT H* "
        )
        if len(sidechain) == 0:
            continue  # skip no sidechain
        centroids.append(sidechain.positions.mean(axis=0))
        resids.append(i)

    return np.array(resids), np.array(centroids)


def compute_centroid_distance_matrix(centroids):
    return distance_array(centroids, centroids)


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
        resids, centroids = compute_sidechain_centrorids(traj_uni, nres)
        mat = compute_centroid_distance_matrix(centroids)
        all_matrices.append(mat)

        print(f"Progress: {i / len(traj_uni.trajectory) * 100:2f}%", end="\r")
    print("\n")
    all_matrices_array = np.array(all_matrices)
    avg_matrix = np.mean(all_matrices_array, axis=0)

    if output_prefix:
        np.save(f"{output_prefix}_dist_matrices.npy", all_matrices_array)
        np.save(f"{output_prefix}_avg_matrices.npy", avg_matrix)
        np.savetxt(f"{output_prefix}_avg_matrices.dat", avg_matrix, fmt="%.3f")
        np.savetxt(f"{output_prefix}_resids.dat", resids, fmt="%.3f")

    print(f"[DONE] {traj_file} in {time() - start_time:.2f}\n")

    return avg_matrix, all_matrices, resids


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print(f"No files found in {INPUT_DIR} matching {GLOB}.")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")

    avg_matrix, all_matrices, resids = universe_setup(files, NAMED_PDB)
