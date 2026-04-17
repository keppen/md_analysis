# Import necessary libraries
from pathlib import Path
from MDAnalysis.analysis.dihedrals import Dihedral
import MDAnalysis as mda
from sklearn.neighbors import KernelDensity
import numpy as np
from ploting import plot_kde_3d


def compute_torsion_dataframe(universe, nres, atom_selector_fn, verbose=True, stop=-1):
    """
    Build a list of AtomGroups of four atoms per residue via atom_selector_fn,
    run the Dihedral analysis, and return a DataFrame of angles.

    Parameters
    ----------
    ref_universe : MDAnalysis.Universe
        The Universe whose trajectory has already been patched.
    nres : int
        Number of residues (max resid + 1).
    atom_selector_fn : callable
        Function taking (ref_universe, i, nres) and returning
        a list of four AtomGroup selections.
    verbose : bool
        Whether to print progress from Dihedral.run.

    Returns
    -------
    pandas.DataFrame
        Flattened angles shape (n_frames * nres, 1) as column "angle".
    """
    dihedral_groups = []
    for i in range(nres):
        sel = atom_selector_fn(universe, i, nres)
        print(f"INFO: Current selection {atom_selector_fn}")
        if not all(len(group) == 1 for group in sel):
            counts = [len(g) for g in sel]
            raise ValueError(f"Could not find all 4 atoms for angle {i}: {counts}")
        # flatten to one AtomGroup of four atoms
        dihedral_groups.append(sel[0] + sel[1] + sel[2] + sel[3])

    dih = Dihedral(dihedral_groups)
    dih.run(verbose=verbose, stop=stop)
    print(dih.results.angles)
    return dih.results.angles


def generate_grid(grid_limits, resolution):
    """
    Create coordinate arrays and stacked points for a grid.
    """
    slices = [
        slice(min_v, max_v, complex(0, resolution)) for min_v, max_v in grid_limits
    ]
    coords = np.mgrid[tuple(slices)]
    points = np.vstack([c.ravel() for c in coords]).T
    return coords, points


def compute_kde(data, grid_limits, resolution, bandwidth=3, kernel="gaussian"):
    """
    Fit a 3D Gaussian KDE and evaluate on a specified grid.
    Returns the density array and grid coordinates.
    """
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, algorithm="ball_tree")
    kde.fit(data)
    coords, points = generate_grid(grid_limits, resolution)
    log_dens = kde.score_samples(points)
    density = np.exp(log_dens).reshape(coords[0].shape)
    return density, coords


def save_kde(filename, density, coords, grid_limits, resolution, labels, png_file):
    np.savez_compressed(
        filename,
        PDF=density,
        coordinates=coords,
        grid_limits=grid_limits,
        resolution=resolution,
        labels=labels,
        png_file=png_file,
    )


if __name__ == "__main__":
    import sys

    # ——— main script ———
    ref_file = Path(sys.argv[1])
    traj_file = Path(sys.argv[2])

    png_file = traj_file.stem + ".3dkde.png"
    npz_file = traj_file.stem + ".3dkde.npz"
    npy_file = traj_file.stem + ".3dkde.data.npy"

    # Load reference (correct names) and trajectory (coordinates)
    # ref = mda.Universe(ref_file)
    traj = mda.Universe(ref_file, traj_file)
    print(traj.trajectory)

    # Validate matching atom counts
    # if len(ref.atoms) != len(traj.atoms):
    #     raise ValueError("Atom counts don't match")

    # Patch the trajectory into the reference Universe
    # ref.trajectory = traj.trajectory

    # Number of residues
    nres = max(r.resid for r in traj.residues) + 1
    print("nres = ", nres)

    # Define selector functions for each torsion type
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
        # last residue has HO in$stead of next N
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

    stop = -1

    # Compute the three torsion DataFrames
    torsion1 = compute_torsion_dataframe(traj, nres, selector_torsion1, stop=stop)
    torsion2 = compute_torsion_dataframe(traj, nres, selector_torsion2, stop=stop)
    torsion3 = compute_torsion_dataframe(traj, nres, selector_torsion3, stop=stop)

    data = np.concatenate([torsion1, torsion2, torsion3], axis=1)
    print(data)
    print(data.shape)
    np.save(npy_file, data)
    # Concatenate and inspect
    data = np.concatenate(
        [[torsion1.ravel(), torsion2.ravel(), torsion3.ravel()]], axis=1
    ).T
    print(data)
    print(data.shape)
    print("WARNING!  KDE calculataion is done! Remove breaking point")

    grid_limits = [(-180, 180), (-180, 180), (-180, 180)]
    resolution = 90
    labels = [r"$\phi$", r"$\xi$", r"$\chi$"]

    PDF, coords = compute_kde(data, grid_limits, resolution)
    save_kde(npz_file, PDF, coords, grid_limits, resolution, labels, png_file)
    plot_kde_3d(
        PDF,
        coords,
        grid_limits,
        "torsion_distribution",
        labels,
        downsample=1,
        output_file=png_file,
    )
