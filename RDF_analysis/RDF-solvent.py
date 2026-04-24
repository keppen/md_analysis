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
TYPE = sys.argv[4]  # SITE or RESIDUE


# POLYMER_SELECTION = "O OT OA"  # e.g. N, O, H*
# SOLVENT_SELECTION = "H"  # e.g. N, O, H*

POLYMER_SELECTION = None
SOLVENT_SELECTION = None
if "--poly" in sys.argv:
    POLYMER_SELECTION = " ".join(
        sys.argv[sys.argv.index("--poly") + 1 : sys.argv.index("--solv")]
    )  # e.g. N, O, H*

if "--solv" in sys.argv:
    SOLVENT_SELECTION = " ".join(
        sys.argv[sys.argv.index("--solv") + 1 :]
    )  # e.g. N, O, H*

if not POLYMER_SELECTION or not SOLVENT_SELECTION:
    print("INVALID SELECTION")
    exit()

GLOB = "full.xtc"

NBINS = 200
RANGE = (0.0, 10.0)
STEP = 50

# -----------------------


def get_polymer_atoms(u, resid=None):
    if resid:
        sel = f"resid {resid} and name {POLYMER_SELECTION}"
    else:
        sel = f"name {POLYMER_SELECTION}"
    print(sel)
    atoms = u.select_atoms(sel)
    return atoms, sel


def get_solvent_atoms(u):
    sel = f"resname LIG and name {SOLVENT_SELECTION}"
    print(sel)
    atoms = u.select_atoms(sel)
    return atoms, sel


def compute_rdf_per_site(traj_file, geom_tpr, named_pdb):
    print(f"[START] {traj_file}")
    start_time = time()

    u = mda.Universe(geom_tpr, traj_file)
    named_u = mda.Universe(named_pdb)

    nres = max(named_u.residues.resids) + 1
    print(nres)

    polymer_atoms, polymer_sel = get_polymer_atoms(named_u)
    solvent_atoms, solvent_sel = get_solvent_atoms(u)
    print(polymer_atoms)

    if len(polymer_atoms) == 0 or len(solvent_atoms) == 0:
        print("[ERROR] Empty selection")
        return None

    rdf_results = {}
    bins = None

    # --- per-site RDF ---
    for atom in polymer_atoms.atoms:
        label = f"{atom.residue.resname}-{atom.residue.resid}:{atom.name}"

        # single_atom_group = u.atoms[atom.index]
        single_atom_group = u.atoms[[atom.index]]

        rdf = InterRDF(
            single_atom_group,
            solvent_atoms,
            nbins=NBINS,
            range=RANGE,
            verbose=True,
        )

        rdf.run(step=STEP)

        if bins is None:
            bins = rdf.bins

        rdf_results[label] = rdf.rdf

    print(f"[DONE] {traj_file} in {time() - start_time:.2f}s")

    return bins, rdf_results


def compute_rdf_per_residue(traj_file, geom_tpr, named_pdb):
    print(f"[START] {traj_file}")
    start_time = time()

    u = mda.Universe(geom_tpr, traj_file)
    named_u = mda.Universe(named_pdb)

    nres = max(named_u.residues.resids) + 1
    print(nres)

    solvent_atoms, solvent_sel = get_solvent_atoms(u)

    rdf_results = {}
    bins = None

    # --- per-residue RDF ---
    for resid in range(nres):
        polymer_atoms, polymer_sel = get_polymer_atoms(named_u, resid=resid)

        if len(polymer_atoms) == 0 or len(solvent_atoms) == 0:
            print("[ERROR] Empty selection")
            return None

        label = (
            f"{named_u.residues[resid].resname}-{resid}:{polymer_atoms.atoms[0].name}"
        )
        print(f"Computing COM RDF for {label}")

        atom_group = u.atoms[polymer_atoms.indices]

        rdf = InterRDF(
            atom_group,
            solvent_atoms,
            nbins=NBINS,
            range=RANGE,
            verbose=True,
        )

        rdf.run(step=STEP)

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
        result = None
        if TYPE == "SITE":
            result = compute_rdf_per_site(f, GEOM_TPR, NAMED_PDB)
        if TYPE == "RESIDUE":
            result = compute_rdf_per_residue(f, GEOM_TPR, NAMED_PDB)

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

    safe_polymer_selection = (
        POLYMER_SELECTION.replace(":", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("*", "star")
    )
    safe_solvent_selection = (
        SOLVENT_SELECTION.replace(":", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("*", "star")
    )
    for key, rdf in avg_results.items():
        safe_key = key.replace(":", "_").replace("-", "_")
        np.save(
            f"rdf_{TYPE}_POLYMER_{safe_polymer_selection}_SOLVENT_{safe_solvent_selection}_{safe_key}.npy",
            rdf,
        )

    print("[DONE] Per-site RDFs computed and saved.")
