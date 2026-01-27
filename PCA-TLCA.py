from time import time
import sys
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import distance_array

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

import matplotlib.pyplot as plt

from deeptime.decomposition import TICA


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

INPUT_DIR = Path(sys.argv[1])
NAMED_PDB = Path(sys.argv[2])  # topology
MIDDLE_PDB = Path(sys.argv[3])  # PDB with middle structures
RMSD_NPY_FILE = None
DISTANCE_NPY_FILE = None

if "--rmsd-npy" in sys.argv:
    RMSD_NPY_FILE = sys.argv[sys.argv.index("--rmsd-npy") + 1]

if "--dist-npy" in sys.argv:
    DISTANCE_NPY_FILE = sys.argv[sys.argv.index("--dist-npy") + 1]

GLOB = "*.xtc"
THRESHOLD = 4  # for finding minima, depth of minimum in kj/mol
RADIUS_MULTIPLIER = 5  # for finding minima, change
N_RMSDS = 5  # for RMSD calculation, number of middle structures
N_MINIMA = 5  # for finding minima, number of minima
SELECTION = "name N HN O OA C CB* CG*"
OUTPUT_PREFIX = "pca"
N_FRAMES = 5001


# ---------------------------------------------------------------------
# CORE WORKFLOW
# ---------------------------------------------------------------------


def universe_setup(
    traj_files,
    top_file: Path,
    middle_file: Path,
    output_prefix: str = "pca",
    select: str = "name N HN O OA C CB* CG*",
    npy_file=None,
):
    """
    Compute RMSD of trajectory to each middle structure,
    then perform PCA on the RMSD feature space.
    """

    start_time = time()

    traj_uni, middle_uni = _load_universes(top_file, traj_files, middle_file)
    traj_sel, middle_sel = _select_atoms(traj_uni, middle_uni, select)

    total_frames = len(traj_uni.trajectory)
    n_middle = len(middle_uni.trajectory)
    n_traj = total_frames // N_FRAMES

    print(f"Trajectory frames : {total_frames}")
    print(f"Middle structures : {n_middle}")

    rmsd_data_space = _load_or_compute_rmsd(
        traj_sel,
        middle_sel,
        n_middle,
        output_prefix,
        RMSD_NPY_FILE,
        N_RMSDS,
    )
    print(rmsd_data_space.shape)

    traj_sel = traj_uni.select_atoms("name OA")

    dist_data_space = _load_or_compute_distances(
        traj_sel, output_prefix, DISTANCE_NPY_FILE
    ).T
    print(dist_data_space.shape)

    # data_space = np.concatenate([rmsd_data_space[:N_RMSDS], dist_data_space], axis=0)
    data_space = dist_data_space
    # data_space = rmsd_data_space
    print(data_space.shape)

    X_all = data_space.T
    X = [X_all[i * N_FRAMES : (i + 1) * N_FRAMES] for i in range(n_traj)]

    # _apply_time_filter(data_space)

    X_pca, pca = _run_pca(X)

    X_tica, tica = _run_tica(X_pca)

    _plot_free_energy_surface(X_tica, OUTPUT_PREFIX, 200)

    print(f"[DONE] Total runtime: {time() - start_time:.2f} s")


# ---------------------------------------------------------------------
# UNIVERSE SETUP
# ---------------------------------------------------------------------


def _load_universes(top_file, traj_files, middle_file):
    print("[START] Loading universes")
    traj_uni = mda.Universe(top_file, traj_files)
    middle_uni = mda.Universe(top_file, middle_file)
    return traj_uni, middle_uni


def _select_atoms(traj_uni, middle_uni, select):
    traj_sel = traj_uni.select_atoms(select)
    middle_sel = middle_uni.select_atoms(select)

    if traj_sel.n_atoms != middle_sel.n_atoms:
        raise ValueError(
            "Atom selection mismatch between trajectory and middle structures"
        )

    return traj_sel, middle_sel


# ---------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------


def _load_or_compute_rmsd(
    traj_sel,
    middle_sel,
    n_middle,
    output_prefix,
    npy_file,
    n_rmsds,
):
    if npy_file:
        data_space = np.load(npy_file)
        if data_space.size == 0:
            raise ValueError("No data loaded!")
        return data_space

    rmsd_arrays = []

    for i, _ in enumerate(middle_sel.universe.trajectory):
        print(f"RMSD vs middle {i + 1}/{n_rmsds}", end="\r")

        if i == n_rmsds:
            break

        R = rms.RMSD(
            traj_sel,
            middle_sel,
            ref_frame=i,
            center=True,
            superposition=True,
            verbose=True,
        )
        R.run(step=-1)

        rmsd_arrays.append(R.results.rmsd[:, 2])

    print("\nRMSD calculation complete.")

    data_space = np.array(rmsd_arrays)

    if output_prefix:
        np.save(f"rmsd-{output_prefix}.npy", data_space)

    return data_space


def _load_or_compute_distances(
    traj_sel,
    output_prefix,
    npy_file,
):
    if npy_file:
        data_space = np.load(npy_file)
        if data_space.size == 0:
            raise ValueError("No data loaded!")
        return data_space

    distance_frames = []

    universe = traj_sel.universe
    n_atoms = len(traj_sel)

    # Precompute index pairs (n, n+j) to avoid duplicates and self-pairs
    pair_indices = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]

    n_frames = len(universe.trajectory)

    for i, ts in enumerate(universe.trajectory):
        print(f" {i + 1}/{n_frames}", end="\r")

        coords = traj_sel.positions

        # Compute full distance matrix for this frame
        dist_matrix = distance_array(coords, coords)

        # Extract only (n, n+j) distances
        frame_distances = np.array([dist_matrix[i, j] for i, j in pair_indices])

        distance_frames.append(frame_distances)

    print("\nC–C distance calculation complete.")

    data_space = np.array(distance_frames)

    if output_prefix:
        np.save(f"cdist-{output_prefix}.npy", data_space)

    return data_space


# ---------------------------------------------------------------------
# FILTERING
# ---------------------------------------------------------------------


def _apply_time_filter(data_space):
    print("Applying Savitzky–Golay filtering...", end="\r")

    for i in range(data_space.shape[0]):
        data_space[i] = savgol_filter(
            data_space[i],
            window_length=100,
            polyorder=6,
        )

    print("done.")


# ---------------------------------------------------------------------
# PCA and ITCA
# ---------------------------------------------------------------------


def _run_pca(X):
    """
    PCA used strictly for noise filtering and decorrelation.
    X : list of arrays, each (n_frames_i, n_features)
    """
    print("Running PCA (noise filtering)...", end="\r")

    # concatenate all trajectories
    X_concat = np.vstack(X)

    # robust scaling (fit once)
    scaler = RobustScaler()
    X_scaled_concat = scaler.fit_transform(X_concat)

    # PCA for denoising
    pca = PCA(n_components=0.95, svd_solver="full", whiten=False)
    X_pca_concat = pca.fit_transform(X_scaled_concat)

    # split back into trajectories
    X_pca = []
    start = 0
    for traj in X:
        n = traj.shape[0]
        X_pca.append(X_pca_concat[start : start + n])
        start += n

    print(
        f"done. PCA retained {X_pca_concat.shape[1]} components "
        f"({pca.explained_variance_ratio_.sum():.3f} variance)"
    )

    return X_pca, pca


def _run_tica(X_pca):
    """
    Run tICA on PCA-denoised trajectories.
    """
    print("Running tICA...", end="\r")

    tica = TICA(
        lagtime=10,  # OK for short trajectories; tune later
        dim=min(10, X_pca[0].shape[1]),
    )
    tica.fit(X_pca)
    X_tica = tica.transform(X_pca)

    print("done.")
    return X_tica, tica


# ---------------------------------------------------------------------
# PLOTTING: FREE ENERGY SURFACE
# ---------------------------------------------------------------------


def _plot_free_energy_surface(X_tica, output_prefix, bins=200, T=300):
    """
    Free energy surface in tICA space (tIC1, tIC2).

    X_tica : list of arrays, each (n_frames_i, n_tics)
    """
    # concatenate trajectories
    X = np.vstack(X_tica)

    x = X[:, 0]
    y = X[:, 1]

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    H[H == 0] = np.nan

    R = 8.3145e-3  # kJ/mol/K
    F = -R * T * np.log(H)

    F -= np.nanmin(F)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        F.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="nipy_spectral",
    )

    fig.colorbar(im, label="Free energy (kJ/mol)")
    ax.set_xlabel("tIC1")
    ax.set_ylabel("tIC2")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}-FES-tICA.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print(f"No files found in {INPUT_DIR} matching {GLOB}")
        sys.exit(1)

    print(f"Found {len(files)} trajectory files.")

    universe_setup(
        files,
        NAMED_PDB,
        MIDDLE_PDB,
        npy_file=RMSD_NPY_FILE,
        select=SELECTION,
        output_prefix=OUTPUT_PREFIX,
    )
