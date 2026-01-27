from time import time
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from numpy.typing import ArrayLike
from pandas import DataFrame
from scipy.spatial import KDTree
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# ---- USER SETTINGS ----

INPUT_DIR = Path(sys.argv[1])
NAMED_PDB = Path(sys.argv[2])
GLOB = "*.xtc"
PROBE_RADIUS_A = 1.4  # Angstrem

# -----------------------

PROBE_RADIUS_NM = PROBE_RADIUS_A / 10  # nm


def generate_sphere_points(n_points: int = 960):
    "Golden spiral algorithm"
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.vstack(
        (
            x,
            y,
            z,
        )
    ).T


def shrake_rupley_sasa(
    sphere_points,
    coords_total,
    radii_total,
    coords_fragment,
    radii_fragment,
    fragment_global_indices: list[int],
    use_kdtree=True,
):
    n_points: int = sphere_points.shape[0]
    sasa_total: float = 0.0
    N_total: int = coords_total.shape[0]
    max_radii_total: float = np.max(radii_total)

    tree = KDTree(coords_total)

    # Fro every fragment atom compute exposed points w.r.t its relevant neighbors set
    for i_frag, (coord, radius_i) in enumerate(zip(coords_fragment, radii_fragment)):
        # translate sphere onto an atom
        surface_points = coord + radius_i * sphere_points

        # get neighbor cantidate list (global indices into coords tootal)
        # query a safe superset radius: radius_i + max_radii_total
        candidate_idx = tree.query_ball_point(coord, r=radius_i + max_radii_total)
        if not candidate_idx:
            # no neighbors -> all points exposed
            n_exposed = n_points
            area = 4.0 * np.pi * (radius_i**2) * (n_exposed / n_points)
            sasa_total += area
            continue

        candidate_idx = np.array(candidate_idx, dtype=int)
        # exlude self if present (map fragment atom to its global index)
        gi = fragment_global_indices[i_frag]
        # filter out self index if appears
        mask_self = candidate_idx != gi
        candidate_idx = candidate_idx[mask_self]
        if candidate_idx.size == 0:
            # no occluders
            n_exposed = n_points
            area = 4.0 * np.pi * (radius_i**2) * (n_exposed / n_points)
            sasa_total += area
            continue

        # refine candidates: obly keep those where center distance < radius_i + radii_total[j]
        neighbor_coords = coords_total[candidate_idx]  # (M,3)
        center_dist = np.linalg.norm(neighbor_coords - coord, axis=1)
        neighbor_radii = radii_total[candidate_idx]
        keep_mask = center_dist < (radius_i + neighbor_radii)
        if not np.any(keep_mask):
            # no occluders after struct test
            n_exposed = n_points
            area = 4.0 * np.pi * (radius_i**2) * (n_exposed / n_points)
            sasa_total += area
            continue

        # keep filtered neighbor coords/radii
        neighbor_coords = neighbor_coords[keep_mask]  # (M2, 3)
        neighbor_radii = neighbor_radii[keep_mask]  # (M2, )

        # compute squared distances surface_points < neigh_radii**2
        # vectorized: (n_points, M2)
        # substract neighbor coords from each surface poin
        # d2 = ((surface_points[:,None,:] - neigh_coords[None,:,:]**2).sum(axis=2)
        d = (
            surface_points[:, None, :] - neighbor_coords[None, :, :]
        )  # (n_points, M2, 3]
        d2 = np.einsum("ijk,ijk->ij", d, d)

        r2 = (neighbor_radii**2)[None, :]  # (1, M2)
        occluded = np.any(d2 < r2, axis=1)
        n_exposed = np.count_nonzero(~occluded)

        # surface area for each point
        area = 4.0 * np.pi * (radius_i**2) * (n_exposed / n_points)
        sasa_total += area

    return sasa_total


def universe_setup(
    traj_file,
    top_file: Path,
    selection: None | str = None,
    probe_radius: float = 1.4,
    n_sphere_points=960,
):
    start_time = time()
    print(f"[START] Loading {traj_file}")

    vdw_radii = {"H": 1.0, "C": 1.7, "N": 1.45, "O": 1.35, "S": 1.8}

    traj_uni = mda.Universe(top_file, traj_file)
    nres: int = max(r.resid for r in traj_uni.residues) + 1

    sel_total: mda.AtomGroup = traj_uni.select_atoms(f"resid 0-{nres}")
    sel_fragment: mda.AtomGroup = (
        traj_uni.select_atoms(selection) if selection else sel_total
    )
    print(f"Selection: {str(selection)}")

    elements_total: list[str] = [
        a.element if a.element else a.name[0] for a in sel_total
    ]
    elements_fragment: list[str] = [
        a.element if a.element else a.name[0] for a in sel_fragment
    ]

    radii_total = (
        np.array([vdw_radii.get(el, 1.7) for el in elements_total]) + probe_radius
    )
    radii_fragment = (
        np.array([vdw_radii.get(el, 1.7) for el in elements_fragment]) + probe_radius
    )

    sphere_points = generate_sphere_points(n_sphere_points)

    fragment_global_indices = sel_fragment.indices

    sasa_values = []
    n_frames = len(traj_uni.trajectory)
    print(f"Processing {n_frames} frames.")
    for i, ts in enumerate(traj_uni.trajectory[::50]):
        coords_total = sel_total.atoms.positions.copy()
        coords_fragment = sel_fragment.atoms.positions.copy()

        sasa_frame = shrake_rupley_sasa(
            sphere_points,
            coords_total,
            radii_total,
            coords_fragment,
            radii_fragment,
            fragment_global_indices,
        )
        sasa_values.append(sasa_frame)
        # print(f"Frame {ts.time}: SASA ratio = {sasa_frame:.5f} Angstrem^2")

    sasa_array = np.array(sasa_values)

    print(f"[DONE] {traj_file} in {time() - start_time:.2f}\n")
    # basic stats
    mean_sasa = np.mean(sasa_array)
    std_sasa = np.std(sasa_array)
    min_idx = int(np.argmin(sasa_array))
    max_idx = int(np.argmax(sasa_array))
    print("\n=== SASA summary ===")
    print(
        f"Frames: {len(sasa_array)}, mean = {mean_sasa:.2f} Å^2, std = {std_sasa:.2f}"
    )
    print(f"Min SASA = {sasa_array[min_idx]:.2f} Å^2 at frame {min_idx}")
    print(f"Max SASA = {sasa_array[max_idx]:.2f} Å^2 at frame {max_idx}")

    return sasa_array


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print(f"No files found in {INPUT_DIR} matching {GLOB}.")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")

    # sasa_traj = universe_setup(files, NAMED_PDB, selection="name N C O OA HN")
    # DataFrame(sasa_traj).to_csv("sasa_urethane.csv")
    sasa_traj = universe_setup(files, NAMED_PDB)
    DataFrame(sasa_traj).to_csv("sasa_total.csv")
