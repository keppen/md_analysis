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
SET_NRES = None

if "--nres" in sys.argv:
    SET_NRES = int(sys.argv[sys.argv.index("--nres") + 1])

GLOB = "full.xtc"

STEP = 50
CUTOFF = 3.6  # Å


# -----------------------


def get_HN_donor_axis(u):
    h_sel = "name HN"
    n_sel = "name N"

    return u.select_atoms(n_sel)[0], u.select_atoms(h_sel)[0]


def get_OH_donor_axis(u):
    h_sel = "name HO"
    n_sel = "name OA"

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
    """return N atom, C_met atom"""
    return residue.atoms[5], residue.atoms[0]


def angle_between(v1, v2):
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1n, v2n)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def mic_vector(pos, origin, box):
    delta = pos - origin
    for i in range(3):
        if box[i] > 0:
            delta[i] -= box[i] * np.round(delta[i] / box[i])
    return delta


def rotation_matrix_from_vectors(vec1, vec2):
    """Find rotation matrix that aligns vec1 to vec2, Rodrigues rotation"""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s == 0:
        return np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))


def build_local_frame(v1, v2):
    """
    v1 → primary axis (z)
    v2 → secondary direction (used to define x)
    """

    z = v1 / np.linalg.norm(v1)

    # remove projection of v2 onto z (Gram-Schmidt)
    x = v2 - np.dot(v2, z) * z
    x /= np.linalg.norm(x)

    y = np.cross(z, x)

    # rotation matrix: columns = new basis
    R = np.stack([x, y, z], axis=1)

    return R


def compute_solvent_orientation(traj_file, geom_tpr, named_pdb, resid):
    print(f"[START ANGLES] {traj_file}")

    u = mda.Universe(geom_tpr, traj_file)
    named_u = mda.Universe(named_pdb)

    polymer_residue = get_residue(named_u, resid)
    if AXIS_TYPE == "AROM":
        atom1, atom2 = get_aromring_axis(polymer_residue)
    elif AXIS_TYPE == "HBOND":
        atom1, atom2 = get_HN_donor_axis(polymer_residue)
    elif AXIS_TYPE == "OHGROUP":
        atom1, atom2 = get_OH_donor_axis(polymer_residue)
    else:
        print("ERROR: Bad axis type selection")
        exit()

    solvent_atoms = get_solvent(u)

    if len(polymer_residue) == 0 or len(solvent_atoms) == 0:
        print("[ERROR] Empty selection")
        return None

    all_positions = []
    all_vectors = []

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

        origin = u.atoms[atom1.index].position

        # rotation to align urethane vector with z-axis
        # R = rotation_matrix_from_vectors(urethane_vector, np.array([0.0, 0.0, 1.0]))

        # define second vector (example: N → C)
        carbon_atom = None
        for a in u.atoms[atom1.index].bonded_atoms.select_atoms("name C*"):
            carbon_atom = a
            if len(a.bonded_atoms) == 3:
                carbon_atom = a
                break
        if not carbon_atom:
            print("No carbon atom refrence was not selected.")
            exit()

        v2 = carbon_atom.position - u.atoms[atom1.index].position

        R = build_local_frame(urethane_vector, v2)

        seen_residues = set()

        for (_, j_solv), dist in zip(pairs, distances):
            res = solvent_atoms[j_solv].residue

            if res.ix in seen_residues:
                continue
            seen_residues.add(res.ix)

            acn_n_atom, acn_c_met_atom = get_solvent_dipole_atoms(res)

            # --- positions relative to urethane N ---
            pos_N = mic_vector(acn_n_atom.position, origin, box)
            pos_C = mic_vector(acn_c_met_atom.position, origin, box)

            # rotate into common frame
            # pos_N_rot = R @ pos_N
            # pos_C_rot = R @ pos_C
            pos_N_rot = R.T @ pos_N
            pos_C_rot = R.T @ pos_C

            solvent_vector = pos_C_rot - pos_N_rot

            all_positions.append(pos_N_rot)
            all_vectors.append(solvent_vector)

    return np.array(all_positions), np.array(all_vectors)


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print("No trajectory files found")
        sys.exit(1)

    named_u = mda.Universe(NAMED_PDB)
    nres = max(named_u.residues.resids) + 1

    all_results = {resid: {"positions": [], "vectors": []} for resid in range(nres)}

    for f in files:
        if SET_NRES:
            result = compute_solvent_orientation(f, GEOM_TPR, NAMED_PDB, SET_NRES)

            if result is None:
                continue

            positions, vectors = result

            all_results[SET_NRES]["positions"].extend(positions)
            all_results[SET_NRES]["vectors"].extend(vectors)

        else:
            for resid in range(nres):
                result = compute_solvent_orientation(f, GEOM_TPR, NAMED_PDB, resid)

                if result is None:
                    continue

                positions, vectors = result

                all_results[resid]["positions"].extend(positions)
                all_results[resid]["vectors"].extend(vectors)
                if resid == 3 and AXIS_TYPE == "HBOND":
                    AXIS_TYPE = "OHGROUP"
                    all_results[4] = {"positions": [], "vectors": []}

                    result = compute_solvent_orientation(f, GEOM_TPR, NAMED_PDB, resid)

                    if result is None:
                        continue

                    positions, vectors = result

                    all_results[resid + 1]["positions"].extend(positions)
                    all_results[resid + 1]["vectors"].extend(vectors)

    # --- save ---
    OUTPUT_FILE = "solvent_vectors_data.npz"

    save_dict = {}

    for resid, data in all_results.items():
        pos = np.array(data["positions"])
        vec = np.array(data["vectors"])

        if len(pos) == 0:
            continue

        save_dict[f"pos_{resid}"] = pos
        save_dict[f"vec_{resid}"] = vec

    np.savez_compressed(OUTPUT_FILE, **save_dict)

    print(f"[SAVE] Data written to {OUTPUT_FILE}")
