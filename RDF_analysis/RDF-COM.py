from time import time
import MDAnalysis as mda
import numpy as np
import sys
from pathlib import Path
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

# ---- USER SETTINGS ----
INPUT_DIR = Path(sys.argv[1])
GEOM_TPR = Path(sys.argv[2])
NAMED_PDB = Path(sys.argv[3])

POLYMER_SELECTION = "CZ* CF* CI* CK"  # e.g. N, O, H*
# POLYMER_SELECTION = sys.argv[3]  # e.g. N, O, H*


GLOB = "full.xtc"

NBINS = 200
RANGE = (0.0, 10.0)
STEP = 100

# -----------------------


def get_polymer_atoms(u, resid):
    sel = f"resid {resid} and name {POLYMER_SELECTION}"
    print(sel)
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
        print(f"PROGRESS: {ts} : {len(u.trajectory)}", end="\r")
        comA = groupA.center_of_mass()

        # compute COMs of all solvent molecules
        comB = np.array([res.atoms.center_of_mass() for res in solvent_residues])

        # distances from A to all solvent COMs
        dists = np.linalg.norm(comB - comA, axis=1)

        hist += np.histogram(dists, bins=edges)[0]

        total_pairs += len(dists)
        n_frames += 1

    # bin centers
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]

    # box volume (assumes constant box)
    vol = u.dimensions[:3].prod()
    density = total_pairs / (n_frames * vol)

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

        label = f"{named_u.residues[resid].resname}-{resid}:COM"
        print(f"Computing COM RDF for {label}")

        groupA = u.atoms[polymer_atoms.indices]
        # print(polymer_atoms.indices)
        # for i, _ in enumerate(polymer_atoms):
        #     print(groupA.atoms[i], " - ", polymer_atoms.atoms[i])

        r, rdf = compute_com_rdf_multi(
            groupA, solvent_residues, u, nbins=NBINS, r_range=RANGE, step=STEP
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

    safe_selection = (
        POLYMER_SELECTION.replace(":", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("*", "star")
    )
    for key, rdf in avg_results.items():
        safe_key = key.replace(":", "_").replace("-", "_")
        np.save(f"rdf_COM{safe_selection}_SOLVCOM_{safe_key}.npy", rdf)

    print("[DONE] Per-site RDFs computed and saved.")
