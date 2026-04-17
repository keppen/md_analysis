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

MODE = sys.argv[3]  # DONOR or ACCEPTOR
SOLVENT_ATOM_NAME = sys.argv[4]  # e.g. N, O, H*

GLOB = "full.xtc"

NBINS = 200
RANGE = (0.0, 10.0)

# -----------------------


def get_polymer_atoms(u, mode):
    hbonds = HydrogenBondAnalysis(universe=u)

    if mode == "DONOR":
        sel = hbonds.guess_donors("resname UNK")
    elif mode == "ACCEPTOR":
        sel = hbonds.guess_acceptors("resname UNK")
    else:
        raise ValueError("MODE must be DONOR or ACCEPTOR")

    atoms = u.select_atoms(sel)
    return atoms, sel


def get_solvent_atoms(u):
    sel = f"resname LIG and name {SOLVENT_ATOM_NAME}"
    atoms = u.select_atoms(sel)
    return atoms, sel


def compute_rdf_per_site(traj_file, geom_tpr):
    print(f"[START] {traj_file}")
    start_time = time()

    u = mda.Universe(geom_tpr, traj_file)

    polymer_atoms, polymer_sel = get_polymer_atoms(u, MODE)
    solvent_atoms, solvent_sel = get_solvent_atoms(u)

    print(f"Polymer ({MODE}): {polymer_sel}")
    print(f"Solvent: {solvent_sel}")
    print(f"Counts → polymer: {len(polymer_atoms)}, solvent: {len(solvent_atoms)}")

    if len(polymer_atoms) == 0 or len(solvent_atoms) == 0:
        print("[ERROR] Empty selection")
        return None

    rdf_results = {}
    bins = None

    # --- per-site RDF ---
    for atom in polymer_atoms:
        label = f"{atom.resname}-{atom.resid}:{atom.name}"
        print(f"Computing RDF for {label}")

        single_atom_group = atom.universe.atoms[[atom.index]]

        rdf = InterRDF(
            single_atom_group,
            solvent_atoms,
            nbins=NBINS,
            range=RANGE,
            verbose=True,
        )

        rdf.run(step=10)

        if bins is None:
            bins = rdf.bins

        rdf_results[label] = rdf.rdf

    print(f"[DONE] {traj_file} in {time() - start_time:.2f}s")

    return bins, rdf_results


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print("No trajectory files found")
        sys.exit(1)

    all_results = {}

    for f in files:
        result = compute_rdf_per_site(f, GEOM_TPR)
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

    for key, rdf in avg_results.items():
        safe_key = key.replace(":", "_").replace("-", "_")
        np.save(f"rdf_{MODE}_{SOLVENT_ATOM_NAME}_{safe_key}.npy", rdf)

    print("[DONE] Per-site RDFs computed and saved.")
