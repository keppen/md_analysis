from time import time
import MDAnalysis as mda
import numpy as np
import sys
from pathlib import Path
from MDAnalysis.lib.distances import capped_distance

# ---- USER SETTINGS ----
INPUT_DIR = Path(sys.argv[1])
GEOM_TPR = Path(sys.argv[2])
NAMED_PDB = Path(sys.argv[3])

GLOB = "full.xtc"

STEP = 20
CUTOFF = 5.5  # Å

# -----------------------


def get_HBDonor_axis(u):
    h_sel = "name HN"
    n_sel = "name N"

    return u.select_atoms(n_sel)[0], u.select_atoms(h_sel)[0]


def get_aromring_axis(u):
    cz_sel = "name CZ"
    ck_sel = "name CK"

    return u.select_atoms(cz_sel)[0], u.select_atoms(ck_sel)[0]


def get_residue(u, resid):
    sel = f"resid {resid}"

    return u.select_atoms(sel)


def get_solvent(u):
    sel = f"resname LIG and name N"
    print(sel)
    atoms = u.select_atoms(sel)
    return atoms


def get_solvent_dipole_atoms(residue):
    return residue.atoms[5], residue.atoms[0]


def refidx2trajatom(u, atom):
    return u.atoms[atom.index]


def angle_between(v1, v2):
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1n, v2n)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def compute_solvent_orientation(traj_file, geom_tpr, named_pdb, resid):
    print(f"[START ANGLES] {traj_file}")

    u = mda.Universe(geom_tpr, traj_file)
    named_u = mda.Universe(named_pdb)

    polymer_residue = get_residue(named_u, resid)
    cz_atom, ck_atom = get_aromring_axis(polymer_residue)
    n_atom, h_atom = get_HBDonor_axis(polymer_residue)

    solvent_atoms = get_solvent(u)

    if len(polymer_residue) == 0 or len(solvent_atoms) == 0:
        print("[ERROR] Empty selection")
        return None

    all_theta1 = []
    all_theta2 = []
    all_dist = []

    for ts in u.trajectory[::STEP]:
        print(f"Progress: {ts.time}", end="\r")
        box = ts.dimensions

        h_position = refidx2trajatom(u, h_atom).position
        n_position = refidx2trajatom(u, n_atom).position
        cz_position = refidx2trajatom(u, cz_atom).position
        ck_position = refidx2trajatom(u, ck_atom).position

        # --- find neighbors using capped distance (fast + PBC aware) ---
        pairs, distances = capped_distance(
            refidx2trajatom(u, h_atom).position[np.newaxis, :],
            solvent_atoms.positions,
            max_cutoff=CUTOFF,
            box=box,
        )

        urethane_hbond_vector = h_position - n_position
        urethane_arom_vector = ck_position - cz_position

        seen_residues = set()
        for (_, j_solv), dist in zip(pairs, distances):
            res = solvent_atoms[j_solv].residue

            if res.ix in seen_residues:
                continue
            seen_residues.add(res.ix)

            acn_n_atom, acn_c_met_atom = get_solvent_dipole_atoms(res)

            solvent_vector = acn_c_met_atom.position - acn_n_atom.position

            theta1 = angle_between(solvent_vector, urethane_hbond_vector)
            theta2 = angle_between(solvent_vector, urethane_arom_vector)

            all_theta1.append(np.degrees(theta1))
            all_theta2.append(np.degrees(theta2))
            all_dist.append(dist)

    return (
        np.array(all_theta1),
        np.array(all_theta2),
        np.array(all_dist),
    )


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_orientation(
    dist,
    theta_hb,
    theta_arom,
    title="Orientation landscape",
    save=None,
    max_points=50000,
):
    """
    3D plot of:
        x = distance
        y = HBOND angle
        z = AROM angle
    """

    dist = np.asarray(dist)
    theta_hb = np.asarray(theta_hb)
    theta_arom = np.asarray(theta_arom)

    # --- optional subsampling for speed ---
    if len(dist) > max_points:
        idx = np.random.choice(len(dist), max_points, replace=False)
        dist = dist[idx]
        theta_hb = theta_hb[idx]
        theta_arom = theta_arom[idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(dist, theta_hb, theta_arom, c=dist, cmap="viridis", s=2, alpha=0.5)

    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("θ_HB (deg)")
    ax.set_zlabel("θ_AROM (deg)")
    ax.set_title(title)

    cb = plt.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label("Distance (Å)")

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print("No trajectory files found")
        sys.exit(1)

    named_u = mda.Universe(NAMED_PDB)
    nres = max(named_u.residues.resids) + 1

    all_results = {
        resid: {"theta1": [], "theta2": [], "dist": []} for resid in range(nres)
    }

    for f in files:
        for resid in range(nres):
            result = compute_solvent_orientation(f, GEOM_TPR, NAMED_PDB, resid)

            if result is None:
                continue

            theta1, theta2, dist = result

            all_results[resid]["theta1"].extend(theta1)
            all_results[resid]["theta2"].extend(theta2)
            all_results[resid]["dist"].extend(dist)

    # --- save ---
    # --- convert to numpy ---

    for resid, data in all_results.items():
        theta_hb = np.array(data["theta1"])
        theta_arom = np.array(data["theta2"])
        dist = np.array(data["dist"])

        if len(theta_hb) == 0:
            continue

        H, xedges, yedges = np.histogram2d(
            theta_hb,
            theta_arom,
            bins=[180, 180],
            range=[[0, 180], [0, 180]],
            density=True,
        )

        import matplotlib.pyplot as plt

        plt.imshow(
            H.T,
            origin="lower",
            aspect="auto",
            extent=[0, 180, 0, 180],
        )

        plt.xlabel("θ_HB (deg)")
        plt.ylabel("θ_AROM (deg)")
        plt.colorbar(label="Probability")

        plt.savefig(f"joint_orientation_resid_{resid}.png", dpi=300)
        plt.close()

        # optional save
        np.save(f"hist_resid_{resid}.npy", H)

    # --- save ---
    # np.save("angle_distance_hist.npy", H)
    # np.save("angle_distance_r_edges.npy", r_edges)
    # np.save("angle_distance_theta_edges.npy", theta_edges)

    print("[DONE] Histogram saved")

    print("[DONE] Angles and distances computed and saved.")
