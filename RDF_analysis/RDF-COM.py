from time import time
import MDAnalysis as mda
import numpy as np
import sys
from pathlib import Path
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.transformations import unwrap

# ---- USER SETTINGS ----
INPUT_DIR = Path(sys.argv[1])
GEOM_TPR = Path(sys.argv[2])
NAMED_PDB = Path(sys.argv[3])
SOLVENT_FILE = Path(sys.argv[4])

POLYMER_SELECTION = "CZ* CF* CI* CK"  # e.g. N, O, H*
# POLYMER_SELECTION = sys.argv[3]  # e.g. N, O, H*


GLOB = "full.xtc"

NBINS = 200
RANGE = (0.0, 10.0)
STEP = 50

# -----------------------


def get_polymer_atoms(u, resid):
    sel = f"resid {resid} and name {POLYMER_SELECTION}"
    atoms = u.select_atoms(sel)
    return atoms, sel


def get_solvent_atoms(u):
    sel = f"resname LIG"
    atoms = u.select_atoms(sel)
    return atoms, sel


def compute_com_rdf_multi(
    groupA, solvent_residues, u, nbins=200, r_range=(0.0, 10.0), step=10
):
    r_min, r_max = r_range
    edges = np.linspace(r_min, r_max, nbins + 1)
    hist = np.zeros(nbins)

    n_frames = 0
    total_pairs = 0  # for normalization

    for ts in u.trajectory[::step]:
        print(ts.time, end="\r")
        # for res in solvent_residues:
        #     unwrap(res.atoms)
        # unwrap(groupA)

        comA = groupA.center_of_mass()

        # compute COMs of all solvent molecules
        comB = np.array([res.atoms.center_of_mass() for res in solvent_residues])

        # distances from A to all solvent COMs
        dists = distance_array(comA.reshape(1, 3), comB, box=ts.dimensions).flatten()
        # print("min:", np.min(dists), "mean:", np.mean(dists))

        hist += np.histogram(dists, bins=edges)[0]

        total_pairs += len(dists)
        n_frames += 1

    # bin centers
    print("Done")
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]

    shell_vol = 4.0 * np.pi * r**2 * dr

    rdf = hist / (n_frames * density * shell_vol)

    return r, rdf


def compute_rdf_per_site(traj_file, geom_tpr, named_pdb):
    print(f"[START] {traj_file}")
    start_time = time()

    u = mda.Universe(geom_tpr, traj_file)
    named_u = mda.Universe(named_pdb)

    nres = max(named_u.residues.resids) + 1
    print(nres)

    u.trajectory.add_transformations(unwrap(u.atoms))

    solvent_atoms, solvent_sel = get_solvent_atoms(u)

    solvent_residues = solvent_atoms.residues

    rdf_results = {}
    bins = None

    # --- per-site COM RDF (polymer atom → solvent molecule COMs) ---
    for resid in range(nres):
        polymer_atoms, polymer_sel = get_polymer_atoms(named_u, resid=resid)

        if len(polymer_atoms) == 0 or len(solvent_atoms) == 0:
            print("[ERROR] Empty selection")
            return None

        label = f"{polymer_atoms.resnames[resid]}-{resid}:COM"
        print(f"Computing COM RDF for {label}")

        groupA = u.atoms[polymer_atoms.indices]
        print(groupA)
        print(solvent_residues)

        r, rdf = compute_com_rdf_multi(
            groupA,
            solvent_residues,
            u,
            nbins=NBINS,
            r_range=RANGE,
            step=STEP,
        )

        if bins is None:
            bins = r

        rdf_results[label] = rdf

    print(f"[DONE] {traj_file} in {time() - start_time:.2f}s")

    return bins, rdf_results


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print("No trajectory files found")
        sys.exit(1)

    # --- compute density from structure ---
    ref = mda.Universe(SOLVENT_FILE)

    solvent_sel = "resname LIG"
    solvent_atoms = ref.select_atoms(solvent_sel)
    solvent_residues = solvent_atoms.residues

    N = len(solvent_residues)

    box = ref.dimensions[:3]
    V = box[0] * box[1] * box[2]

    density = N / V  # molecules / A^3
    # ---

    all_results = {}

    for f in files:
        result = compute_rdf_per_site(f, GEOM_TPR, NAMED_PDB)
        if result is None:
            continue

        bins, rdf_dict = result

        for key, rdf in rdf_dict.items():
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(rdf)

    # --- average across trajectories ---
    avg_results = {}
    for key, rdf_list in all_results.items():
        avg_results[key] = np.mean(rdf_list, axis=0)

    # --- save ---
    np.save("rdf_bins.npy", bins)

    for i, (key, rdf) in enumerate(avg_results.items()):
        safe_key = key.replace(":", "_").replace("-", "_")
        np.save(f"rdf_POLYCOM_SOLVCOM_{i}.npy", rdf)

    print("[DONE] Per-site RDFs computed and saved.")
