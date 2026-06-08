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
AXIS_TYPE = sys.argv[4]
SOLVENT_TYPE = sys.argv[5]

GLOB = "full.xtc"

STEP = 20
CUTOFF = 7.02  # Å

# -----------------------


def get_HBDonor_axis(u):
    h_sel = "name HN"
    n_sel = "name N"

    return u.select_atoms(n_sel)[0], u.select_atoms(h_sel)[0]


def get_HBAcceptor_axis(u):
    c_sel = "name C"
    o_sel = "name O"

    return u.select_atoms(o_sel)[0], u.select_atoms(c_sel)[0]


def get_HBAcceptor_T_axis(u):
    c_sel = "name CT"
    o_sel = "name OT"

    return u.select_atoms(o_sel)[0], u.select_atoms(c_sel)[0]


def get_OH_axis(u):
    h_sel = "name HO"
    n_sel = "name OA"

    return u.select_atoms(n_sel)[0], u.select_atoms(h_sel)[0]


def get_aromring_axis(u):
    cz_sel = "name CZ"
    ck_sel = "name CK"

    print("SELECTING AROMATIC ATOMS")
    return u.select_atoms(cz_sel)[0], u.select_atoms(ck_sel)[0]


def get_residue(u, resid):
    sel = f"resid {resid}"

    return u.select_atoms(sel)


def get_solvent_acn(u):
    sel = f"resname LIG and name N"
    print(sel)
    atoms = u.select_atoms(sel)
    return atoms


def get_solvent_chcl3(u):
    sel = f"resname LIG and name H"
    print(sel)
    atoms = u.select_atoms(sel)
    return atoms


def get_anc_dipole_atoms(residue):
    """return N atom, C_met atom"""
    return residue.atoms[5], residue.atoms[0]


def get_chcl3_dipole_atoms(residue):
    return residue.atoms[1], residue.atoms[2]


def angle_between(v1, v2):
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1n, v2n)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def compute_solvent_orientation(
    traj_file, geom_tpr, named_pdb, resid, selection_function
):
    print(f"[START ANGLES] {traj_file}")

    u = mda.Universe(geom_tpr, traj_file)
    named_u = mda.Universe(named_pdb)

    polymer_residue = get_residue(named_u, resid)
    atom1, atom2 = selection_function(polymer_residue)
    print(polymer_residue)

    if SOLVENT_TYPE == "ACN":
        solvent_atoms = get_solvent_acn(u)
    elif SOLVENT_TYPE == "CHCL3":
        solvent_atoms = get_solvent_chcl3(u)
    else:
        print("[ERROR] Empty solvent selection")
        return None
    print(atom1, atom2)
    print(solvent_atoms)

    if len(polymer_residue) == 0 or len(solvent_atoms) == 0:
        print("[ERROR] Empty selection")
        return None

    all_theta = []
    all_dist = []

    for ts in u.trajectory[::STEP]:
        print(f"Progress: {ts.time}", end="\r")
        box = ts.dimensions

        # --- find neighbors using capped distance (fast + PBC aware) ---
        pairs, distances = capped_distance(
            u.atoms[atom2.index].position[np.newaxis, :],
            solvent_atoms.positions,
            max_cutoff=CUTOFF,
            box=box,
        )

        urethane_vector = u.atoms[atom2.index].position - u.atoms[atom1.index].position

        seen_residues = set()
        for (_, j_solv), dist in zip(pairs, distances):
            res = solvent_atoms[j_solv].residue

            if res.ix in seen_residues:
                continue
            seen_residues.add(res.ix)

            if SOLVENT_TYPE == "ACN":
                solvent_atom1, solvent_atom2 = get_anc_dipole_atoms(res)
            elif SOLVENT_TYPE == "CHCL3":
                solvent_atom1, solvent_atom2 = get_chcl3_dipole_atoms(res)
            else:
                print("[ERROR] Wrong atoms selection")
                return None

            solvent_vector = solvent_atom2.position - solvent_atom1.position

            theta = angle_between(solvent_vector, urethane_vector)

            all_theta.append(np.degrees(theta))
            all_dist.append(dist)

    return (
        np.array(all_theta),
        np.array(all_dist),
    )


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print("No trajectory files found")
        sys.exit(1)

    named_u = mda.Universe(NAMED_PDB)
    nres = max(named_u.residues.resids) + 1

    if AXIS_TYPE == "DONOR":
        all_results = {}
        for f in files:
            for resid in range(nres):
                all_results[resid] = {"angles": [], "dist": []}
                result = compute_solvent_orientation(
                    f, GEOM_TPR, NAMED_PDB, resid, get_HBDonor_axis
                )

                if result is None:
                    continue

                angles, dist = result

                all_results[resid]["angles"].extend(angles)
                all_results[resid]["dist"].extend(dist)

            index = nres + 1
            resid = nres

            all_results[index] = {"angles": [], "dist": []}

            result = compute_solvent_orientation(
                f, GEOM_TPR, NAMED_PDB, resid, get_OH_axis
            )

            if result is None:
                continue

            positions, vectors = result

            all_results[index]["angles"].extend(positions)
            all_results[index]["dist"].extend(vectors)

    if AXIS_TYPE == "ACCEPTOR":
        all_results = {}
        for f in files:
            resid = 0
            all_results[resid] = {"angles": [], "dist": []}

            result = compute_solvent_orientation(
                f, GEOM_TPR, NAMED_PDB, resid, get_HBAcceptor_T_axis
            )

            if result is None:
                continue

            positions, vectors = result

            all_results[resid]["angles"].extend(positions)
            all_results[resid]["dist"].extend(vectors)
            for index in range(1, nres + 1):
                all_results[4] = {"angles": [], "dist": []}
                resid = index - 1
                result = compute_solvent_orientation(
                    f, GEOM_TPR, NAMED_PDB, resid, get_HBAcceptor_axis
                )

                if result is None:
                    continue

                angles, dist = result

                all_results[index]["angles"].extend(angles)
                all_results[index]["dist"].extend(dist)

            index = nres + 2
            resid = 3
            all_results[index] = {"angles": [], "dist": []}

            result = compute_solvent_orientation(
                f, GEOM_TPR, NAMED_PDB, resid, get_OH_axis
            )

            if result is None:
                continue

            positions, vectors = result

            all_results[index]["angles"].extend(positions)
            all_results[index]["dist"].extend(vectors)

    if AXIS_TYPE == "AROM":
        all_results = {}
        for f in files:
            for resid in range(nres):
                all_results[resid] = {"angles": [], "dist": []}
                result = compute_solvent_orientation(
                    f, GEOM_TPR, NAMED_PDB, resid, get_aromring_axis
                )

                if result is None:
                    continue

                angles, dist = result

                all_results[resid]["angles"].extend(angles)
                all_results[resid]["dist"].extend(dist)

    # --- save ---
    # --- convert to numpy ---

    for resid, data in all_results.items():
        angles = np.array(data["angles"])
        dist = np.array(data["dist"])

        if len(angles) == 0:
            continue

        H, r_edges, theta_edges = np.histogram2d(
            dist,
            angles,
            bins=[200, 180],
            range=[[0.0, CUTOFF], [0.0, 180.0]],
            density=True,
        )

        # --- plot ---
        import matplotlib.pyplot as plt

        plt.imshow(
            H.T,
            origin="lower",
            aspect="auto",
            extent=[r_edges[0], r_edges[-1], theta_edges[0], theta_edges[-1]],
        )

        plt.xlabel("Distance (Å)")
        plt.ylabel("Angle (deg)")
        plt.title(f"Resid {resid}")
        plt.colorbar(label="Probability density")

        plt.savefig(f"angle_vs_distance_resid_{resid}.png", dpi=300)
        plt.close()

        # optional save
        np.save(f"hist_resid_{resid}.npy", H)
    # --- save ---
    # np.save("angle_distance_hist.npy", H)
    # np.save("angle_distance_r_edges.npy", r_edges)
    # np.save("angle_distance_theta_edges.npy", theta_edges)

    print("[DONE] Histogram saved")

    print("[DONE] Angles and distances computed and saved.")
