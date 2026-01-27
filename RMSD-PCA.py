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

    n_frames = len(traj_uni.trajectory)
    n_middle = len(middle_uni.trajectory)

    print(f"Trajectory frames : {n_frames}")
    print(f"Middle structures : {n_middle}")

    # rmsd_data_space = _load_or_compute_rmsd(
    #     traj_sel,
    #     middle_sel,
    #     n_middle,
    #     output_prefix,
    #     RMSD_NPY_FILE,
    #     N_RMSDS,
    # )
    # print(rmsd_data_space.shape)

    traj_sel = traj_uni.select_atoms("name OA")

    dist_data_space = _load_or_compute_distances(
        traj_sel, output_prefix, DISTANCE_NPY_FILE
    ).T
    print(dist_data_space.shape)

    # data_space = np.concatenate([rmsd_data_space[:N_RMSDS], dist_data_space], axis=0)
    data_space = dist_data_space
    # data_space = rmsd_data_space
    print(data_space.shape)

    plot_feature_distributions(
        data_space,
        max_features=15,
        title="RMSD and Distance Feature Distributions",
    )

    # _apply_time_filter(data_space)

    reduced_data, explained_variance_ratio, pca = _run_pca(data_space)

    _plot_free_energy_surface(
        reduced_data,
        n_frames,
        output_prefix,
    )

    _plot_explained_variance(
        explained_variance_ratio,
        output_prefix,
    )

    _plot_loadings(
        pca.components_,
        data_space.shape[0],
        output_prefix,
    )

    minima_pca_coords, regions = _find_free_energy_minima(
        reduced_data,
        n_frames,
        THRESHOLD,
        RADIUS_MULTIPLIER,
        N_MINIMA,
    )

    _plot_fes_with_minima(
        reduced_data,
        minima_pca_coords,
        regions,
        output_prefix,
    )

    _save_structure_indices(
        reduced_data,
        regions,
    )

    save_average_structures_per_minimum(
        traj_uni,
        "all",
        reduced_data,
        regions,
        output_prefix=OUTPUT_PREFIX,
    )

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
# PCA
# ---------------------------------------------------------------------


def _run_pca(data_space):
    print("Running PCA...", end="\r")

    X = data_space.T
    scaler = StandardScaler(
        with_std=False
    )  # tlyko odejmowanie średniej. Dzielenie przez odchylenie std. odbiera fizycznosc analizy
    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        whiten=False
    )  # bez normalizowania wariancji. Pozbywa się wtedy fizycznej interpetacji.
    reduced_data = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_

    print("done.")
    print("Explained variance:", explained_variance_ratio[:5])

    return reduced_data, explained_variance_ratio, pca


# ---------------------------------------------------------------------
# PLOTTING: FREE ENERGY SURFACE
# ---------------------------------------------------------------------


def _plot_free_energy_surface(reduced_data, n_frames, output_prefix):
    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    H[H == 0] = np.nan

    T = 300
    kb = 1.380649e-23
    Na = 6.02214076e23
    R = 8.3145

    DG = -Na * kb * T * np.log(H / n_frames) / 1000

    fig, ax = plt.subplots(figsize=(7, 5))
    cax = ax.imshow(
        DG.T,
        origin="lower",
        aspect="auto",
        cmap="nipy_spectral",
    )
    fig.colorbar(cax, label="kJ/mol")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}-FES.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# EXPLAINED VARIANCE
# ---------------------------------------------------------------------


def _plot_explained_variance(explained_variance_ratio, output_prefix):
    plt.figure()
    plt.bar(
        range(1, len(explained_variance_ratio) + 1),
        explained_variance_ratio,
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance ratio")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}-explained.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# LOADINGS
# ---------------------------------------------------------------------


def _plot_loadings(loadings, n_features, output_prefix):
    feature_names = [f"Conf {i}" for i in range(n_features)]

    for pc in (0, 1):
        plt.figure(figsize=(14, 5))
        plt.bar(range(len(loadings[pc])), loadings[pc])
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.ylabel(f"PC{pc + 1} loading")

        plt.tight_layout()
        plt.savefig(f"{output_prefix}-loadings-PC{pc + 1}.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------------
# MINIMA DETECTION
# ---------------------------------------------------------------------


def _find_free_energy_minima(
    reduced_data, n_frames, threshold=10, radius_multiplier=2, n_minima=3
):
    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    H[H == 0] = np.nan

    T = 300
    kb = 1.380649e-23
    Na = 6.02214076e23
    R = 8.3145
    beta = 1 / (R * T / 1000)

    DG = -Na * kb * T * np.log(H / n_frames) / 1000

    valid_mask = H > 0
    DG_masked = np.ma.masked_where(~valid_mask, DG)
    DG_smooth = gaussian_filter(DG_masked, sigma=0.1)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    DG_kJmol = np.ma.masked_where(~valid_mask, DG_smooth.copy())

    minima_pca_coords = []
    regions = []
    global_mask = np.zeros_like(DG_kJmol.mask, dtype=bool)

    def circular_mask(x_center, y_center, radius):
        return (X - x_center) ** 2 + (Y - y_center) ** 2 <= radius**2

    while True:
        DG_tmp = np.ma.masked_where(global_mask | DG_kJmol.mask, DG_kJmol)
        if DG_tmp.count() == 0:
            break

        min_idx = np.unravel_index(np.argmin(DG_tmp), DG_tmp.shape)
        x_min = x_centers[min_idx[0]]
        y_min = y_centers[min_idx[1]]

        minima_pca_coords.append((x_min, y_min))

        for radius in np.linspace(0.03, 0.3, 100):
            mask = circular_mask(x_min, y_min, radius)
            masked_vals = DG_kJmol[mask]

            if masked_vals.count() < 5:
                continue
            if masked_vals.max() - masked_vals.min() > threshold:
                break

        region_mask = circular_mask(x_min, y_min, radius)
        regions.append(region_mask)

        global_mask |= circular_mask(x_min, y_min, radius * radius_multiplier)

        print(
            f"Min {len(minima_pca_coords)} Free energy = "
            f"{masked_vals.min():.2f} kJ/mol at PCA coords "
            f"({x_min:.3f}, {y_min:.3f})"
        )

        if len(minima_pca_coords) == n_minima:
            break

    return minima_pca_coords, regions


# ---------------------------------------------------------------------
# STRUCTURE EXTRACTION
# ---------------------------------------------------------------------


def _save_structure_indices(reduced_data, regions):
    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    x_bin_idx = np.digitize(reduced_data[:, 0], xedges) - 1
    y_bin_idx = np.digitize(reduced_data[:, 1], yedges) - 1

    x_bin_idx = np.clip(x_bin_idx, 0, len(x_centers) - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, len(y_centers) - 1)

    for i, region_mask in enumerate(regions):
        idx = []
        for j in range(reduced_data.shape[0]):
            if region_mask[x_bin_idx[j], y_bin_idx[j]]:
                idx.append(j)

        idx = np.array(idx, dtype=int)
        print(f"Min {i + 1}: Saving {len(idx)} structures.")
        np.save(f"pca-idx-min{i + 1}.npy", idx)


def save_average_structures_per_minimum(
    universe,
    atom_selection,
    reduced_data,
    regions,
    output_prefix,
):
    """
    For each PCA minimum region:
      - collect all structures belonging to that region
      - compute the average structure
      - compute RMSD of each structure to the average
      - save the average structure and RMSD statistics
    """

    # Histogram definition (must match region construction)
    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    x_bin_idx = np.digitize(reduced_data[:, 0], xedges) - 1
    y_bin_idx = np.digitize(reduced_data[:, 1], yedges) - 1

    x_bin_idx = np.clip(x_bin_idx, 0, len(x_centers) - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, len(y_centers) - 1)

    sel = universe.select_atoms(atom_selection)
    n_atoms = sel.n_atoms

    for i, region_mask in enumerate(regions):
        frame_indices = []

        for j in range(reduced_data.shape[0]):
            if region_mask[x_bin_idx[j], y_bin_idx[j]]:
                frame_indices.append(j)

        frame_indices = np.array(frame_indices, dtype=int)
        print(frame_indices)

        if frame_indices.size == 0:
            print(f"Min {i + 1}: no structures found, skipping.")
            continue

        print(f"Min {i + 1}: Averaging {len(frame_indices)} structures.")

        # ------------------------------------------------------------
        # COLLECT COORDINATES
        # ------------------------------------------------------------

        # coords = np.zeros((len(frame_indices), n_atoms, 3), dtype=np.float64)

        # ------------------------------------------------------------
        # REFERENCE STRUCTURE (first frame in minimum)
        # ------------------------------------------------------------

        universe.trajectory[frame_indices[0]]
        ref_atoms = sel.copy()  # AtomGroup copy
        ref_coords = ref_atoms.positions.copy()

        # ------------------------------------------------------------
        # COLLECT ALIGNED COORDINATES
        # ------------------------------------------------------------

        # for k, frame in enumerate(frame_indices):
        #     universe.trajectory[frame]
        #
        #     # Align sel (mobile) to ref_atoms (reference)
        #     align.alignto(
        #         sel,
        #         ref_atoms,
        #         weights=None,
        #     )
        #
        #     coords[k] = sel.positions.copy()

        # ------------------------------------------------------------
        # AVERAGE STRUCTURE
        # ------------------------------------------------------------

        # avg_coords = coords.sum(axis=0) / len(frame_indices)
        # print(avg_coords)

        # ------------------------------------------------------------
        # RMSD TO AVERAGE
        # ------------------------------------------------------------

        # rmsd_vals = np.zeros(len(frame_indices), dtype=np.float64)
        #
        # ref_coords = avg_coords.copy()
        #
        # for k in range(len(frame_indices)):
        #     diff = coords[k] - ref_coords
        #     rmsd_vals[k] = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        #
        # rmsd_mean = rmsd_vals.mean()
        # rmsd_std = rmsd_vals.std()
        #
        # print(f"Min {i + 1}: RMSD to average = {rmsd_mean:.3f} ± {rmsd_std:.3f} Å")

        # ------------------------------------------------------------
        # SAVE AVERAGE STRUCTURE
        # ------------------------------------------------------------

        # universe.trajectory[frame_indices[0]]
        # sel.positions = avg_coords
        #
        # pdb_name = f"{output_prefix}-avg-min{i + 1}.pdb"
        #
        # sel.write(
        #     pdb_name,
        #     bonds="conect",
        # )

        # ------------------------------------------------------------
        # SAVE ALL FRAMES FROM THIS MINIMUM AS ONE MULTI-MODEL PDB
        # ------------------------------------------------------------

        pdb_name = f"{output_prefix}-ensemble-min{i + 1}.pdb"

        with mda.Writer(pdb_name, sel.n_atoms, bonds="conect") as W:
            for k, frame in enumerate(frame_indices):
                if k % 10 != 0:
                    continue

                print(f"Progress: {k} / {len(frame_indices)}", end="\r")
                universe.trajectory[frame]

                # Align to reference again (same as above)
                align.alignto(
                    sel,
                    ref_atoms,
                    weights=None,
                )

                W.write(sel)


# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------


def _plot_fes_with_minima(
    reduced_data,
    minima_pca_coords,
    regions,
    output_prefix,
):
    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    H[H == 0] = np.nan

    T = 300
    kb = 1.380649e-23
    Na = 6.02214076e23

    DG = -Na * kb * T * np.log(H / len(reduced_data)) / 1000
    DG_smooth = gaussian_filter(DG, sigma=0.1)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    plt.figure(figsize=(17 / 2.5, 4))

    plt.contourf(
        x_centers,
        y_centers,
        DG_smooth.T,
        levels=100,
        cmap="nipy_spectral",
    )
    cbar = plt.colorbar(label="Free energy [kJ/mol]")
    cbar.set_label("Free energy [kJ/mol]", rotation=270, labelpad=15)

    for i, (x, y) in enumerate(minima_pca_coords):
        plt.plot(x, y, "wo")
        plt.text(x + 0.03, y + 0.03, f"Min {i + 1}", fontsize=9, weight="bold")

    for (x, y), mask in zip(minima_pca_coords, regions):
        plt.contour(
            x_centers,
            y_centers,
            mask.T.astype(int),
            levels=[0.5],
            colors="white",
            linewidths=1.5,
        )
        plt.plot(x, y, "wo")

    plt.xlabel("Principal Component I")
    plt.ylabel("Principal Component II")
    plt.title("Free Energy Surface with Minima")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}-FES-smoothed.png", dpi=1000)
    plt.close()


def plot_feature_distributions(
    data_space,
    feature_indices=None,
    n_bins=50,
    max_features=20,
    figsize=(10, 6),
    title="Feature Distributions",
):
    """
    Plot distributions (histograms) of selected features.

    Parameters
    ----------
    data_space : np.ndarray
        Shape (n_features, n_samples)

    feature_indices : list[int] or None
        Indices of features to plot. If None, the first `max_features`
        features are plotted.

    n_bins : int
        Number of histogram bins.

    max_features : int
        Maximum number of features to plot if feature_indices is None.

    figsize : tuple
        Matplotlib figure size.

    title : str
        Figure title.
    """
    n_features, n_samples = data_space.shape

    if feature_indices is None:
        feature_indices = list(range(min(n_features, max_features)))

    plt.figure(figsize=figsize)

    for idx in feature_indices:
        values = data_space[idx]

        plt.hist(
            values,
            bins=n_bins,
            density=True,
            alpha=0.5,
            label=f"Feature {idx}",
        )

    plt.xlabel("Value")
    plt.ylabel("Probability density")
    plt.title(title)

    if len(feature_indices) <= 10:
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"data_structure.png", dpi=200)
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
