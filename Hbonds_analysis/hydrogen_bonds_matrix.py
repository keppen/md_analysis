from time import time
import MDAnalysis as mda
import numpy as np
import sys
from pathlib import Path
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- USER SETTINGS ----

INPUT_DIR = Path(sys.argv[1])
NAMED_PDB = Path(sys.argv[2])
TOPOLOGY_ITP = Path(sys.argv[3])
GLOB = "0-classic-*.xtc"
D_A_CUTOFF = 3.5
D_H_A_ANGLE = 155
N_WORKERS = 1
SAVE_PREFIX = "test_hbonds"
# -----------------------


def compute_hbonds(config, chunk):
    name_file = config["name_file"]
    traj_file = config["traj_file"]
    top_file = config["top_file"]
    donors_sel = config["donors_sel"]
    acceptors_sel = config["acceptors_sel"]

    traj_uni = mda.Universe(name_file, traj_file)
    top_uni = mda.Universe(top_file, topology_format="ITP")

    traj_uni.add_TopologyAttr("charges", top_uni.residues[0].atoms.charges)

    nres: int = max(r.resid for r in traj_uni.residues) + 1

    donors = traj_uni.select_atoms(donors_sel)
    donors = donors[np.argsort(donors.resids)]
    acceptors = traj_uni.select_atoms(acceptors_sel)
    acceptors = acceptors[np.argsort(acceptors.resids)]

    if len(donors) == 0 or len(acceptors) == 0:
        print(f"No donors or acceptors found: D={donors}, A={acceptors}")
        exit(1)

    hba = HydrogenBondAnalysis(
        universe=traj_uni,
        donors_sel=donors_sel,
        acceptors_sel=acceptors_sel,
        d_a_cutoff=D_A_CUTOFF,
        d_h_a_angle_cutoff=D_H_A_ANGLE,
        update_selections=True,
    )
    hba.run(start=chunk[0], stop=chunk[1], step=10, verbose=True)

    hb_array = hba.results.hbonds
    hb_matrix = np.zeros([len(donors), len(acceptors)])

    for i, d in enumerate(donors):
        for j, a in enumerate(acceptors):
            match = np.where((hb_array[:, 1] == d.ix) & (hb_array[:, 3] == a.ix))[0]
            if match.size:
                hb_matrix[i, j] = len(match)
    return hb_array, hb_matrix


def universe_setup(
    traj_file, name_file: Path, top_file: Path, output_prefix: str = "hbonds"
):
    """
    Compute sidechain centroid distance matices for all frames.

    Parameters
    traj_file : str, trajectory file name
    top_file : str, topology file name
    output_prefix: str, optional, if given, saves the data in file with a prefix

    Returns
    avg_matrix: np.array
    all_matrices : list of np.ndarray
    resids : np.ndarray
    """

    start_time = time()
    results_matrix = []
    results_array = []
    print(f"[START] Loading {traj_file}")
    traj_uni = mda.Universe(name_file, traj_file)

    nres = max([r.resid for r in traj_uni.residues])
    print(nres)

    donors_sel = f"name N NT or (resid {nres} and name OA)"
    acceptors_sel = "name O OT"
    donors = traj_uni.select_atoms(donors_sel)
    donors = donors[np.argsort(donors.resids)]
    acceptors = traj_uni.select_atoms(acceptors_sel)
    acceptors = acceptors[np.argsort(acceptors.resids)]

    config_hb_analysis = {
        "name_file": name_file,
        "traj_file": traj_file,
        "top_file": top_file,
        "donors_sel": donors_sel,
        "acceptors_sel": acceptors_sel,
    }

    n_frames = len(traj_uni.trajectory)
    print(f"Processing {n_frames} frames.")
    chunk_size = n_frames // N_WORKERS + 1
    chunks = [
        (i, min(i + chunk_size, n_frames)) for i in range(0, n_frames, chunk_size)
    ]
    print(chunks)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(compute_hbonds, config_hb_analysis, chunk): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            traj_file = futures[future]
            try:
                hb_array, hb_matrix = future.result()
                results_array.append(hb_array)
                results_matrix.append(hb_matrix)
            except Exception as e:
                print(f"[ERROR] {traj_file}: {e}")

    if not results_array or not results_matrix:
        print("[ERROR] No valid results.")
        exit(1)

    print(results_array[0].shape)
    stacked_matrix = np.stack(results_matrix)
    stacked_array = np.vstack(results_array)
    print(stacked_array)
    avg_matrix = stacked_matrix.sum(axis=0)
    resids = np.array(
        [
            [f"{a.resname}-{a.resid}:{a.name}" for a in donors],
            [f"{a.resname}-{a.resid}:{a.name}" for a in acceptors],
        ],
        dtype=object,
    )
    print(resids)

    if output_prefix:
        np.save(f"{output_prefix}_avg.npy", avg_matrix)
        np.save(f"{output_prefix}_all_matrix.npy", stacked_matrix)
        np.save(f"{output_prefix}_all_array.npy", stacked_array)
        np.save(f"{output_prefix}_indexes.npy", resids)
        print(f"Data has been saved.")

    print(f"[DONE] {traj_file} in {time() - start_time:.2f}\n")

    return avg_matrix, stacked_matrix, resids


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print(f"No files found in {INPUT_DIR} matching {GLOB}.")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")

    avg_matrix, all_matrices, resids = universe_setup(
        files,
        NAMED_PDB,
        TOPOLOGY_ITP,
    )

    print(resids)
    print(avg_matrix)
