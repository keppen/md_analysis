from time import time
import re
import MDAnalysis as mda
import numpy as np
import sys
from pathlib import Path
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- USER SETTINGS ----

INPUT_DIR = Path(sys.argv[1])
NAMED_PDB = Path(sys.argv[2])
GEOM_TPR = Path(sys.argv[3])
GLOB = "full.xtc"
D_A_CUTOFF = 3.5
D_H_A_ANGLE = 155
N_WORKERS = 30
STEP = 10
SAVE_PREFIX = "solvent-solute"


# -----------------------
def sort_list(lst, str_reverse=False):
    import random

    def get_num(item):
        return int(re.search(r"-(\d+):", item).group(1))

    def get_str(item):
        return re.search(r":([A-Za-z]+)", item).group(1)

    lst = sorted(lst, key=get_num)  # num rosnąco
    lst = sorted(lst, key=get_str, reverse=str_reverse)  # str malejąco
    return lst


def compute_hbonds(config, chunk):
    geom_tpr = config["geom_tpr"]
    traj_trr = config["traj_trr"]
    donors_sel = config["donors_sel"]
    acceptors_sel = config["acceptors_sel"]

    traj_uni = mda.Universe(geom_tpr, traj_trr)

    donors = traj_uni.select_atoms(donors_sel)
    acceptors = traj_uni.select_atoms(acceptors_sel)
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
    hba.run(start=chunk[0], stop=chunk[1], step=STEP, verbose=True)

    hb_array = hba.results.hbonds
    return hb_array


def universe_setup(
    traj_trr,
    name_file: Path,
    geom_tpr: Path,
    output_prefix: str = "hbonds",
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

    print(f"[START] Loading {traj_trr}")
    traj_uni = mda.Universe(geom_tpr, traj_trr, stop=1)

    # Original analysis setup
    hbonds = HydrogenBondAnalysis(
        universe=traj_uni,
        d_a_cutoff=D_A_CUTOFF,
        d_h_a_angle_cutoff=D_H_A_ANGLE,
        update_selections=False,
    )

    update_selections = True
    ureth_donors_sel = hbonds.guess_donors("resname UNK")
    solvent_donors_sel = hbonds.guess_donors("resname LIG")
    ureth_acceptors_sel = hbonds.guess_acceptors("resname UNK")
    solvent_acceptors_sel = hbonds.guess_acceptors("resname LIG")

    donors_sel = f"({ureth_donors_sel})"
    if solvent_donors_sel:
        donors_sel += (
            f" or ({solvent_donors_sel} and around 3.5 resname UNK)"  # Fixed syntax
        )
    acceptors_sel = f"({ureth_acceptors_sel})"
    if solvent_acceptors_sel:
        acceptors_sel += (
            f" or ({solvent_acceptors_sel} and around 3.5 resname UNK)"  # Fixed syntax
        )
    if not solvent_donors_sel and not solvent_acceptors_sel:
        update_selections = False

    config_hb_analysis = {
        "geom_tpr": geom_tpr,
        "traj_trr": traj_trr,
        "donors_sel": donors_sel,
        "acceptors_sel": acceptors_sel,
    }

    print(donors_sel)
    print(acceptors_sel)

    n_frames = len(traj_uni.trajectory)
    print(f"Processing {n_frames} frames.")
    chunk_size = n_frames // N_WORKERS + 1
    chunks = [
        (i, min(i + chunk_size, n_frames)) for i in range(0, n_frames, chunk_size)
    ]
    print(chunks)
    results_array = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(compute_hbonds, config_hb_analysis, chunk): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            traj_trr = futures[future]
            try:
                hb_array = future.result()
                results_array.append(hb_array)
            except Exception as e:
                print(f"[ERROR] {traj_trr}: {e}")

    if not results_array:
        print("[ERROR] No valid results.")
        exit(1)

    stacked_array = np.vstack(results_array)
    print(stacked_array.shape)

    name_uni = mda.Universe(name_file)
    nres = max([r.resid for r in name_uni.residues])
    donors_sel = f"name N NT or (resid {nres} and name OA)"
    # donors_sel = f"name HN HO HTO"
    acceptors_sel = "name O OT"
    donors = name_uni.select_atoms(donors_sel)
    donors = donors[np.argsort(donors.resids)]
    acceptors = name_uni.select_atoms(acceptors_sel)
    acceptors = acceptors[np.argsort(acceptors.resids)]
    resids = np.array(
        [
            [f"{a.resname}-{a.resid}:{a.name}" for a in donors],
            [f"{a.resname}-{a.resid}:{a.name}" for a in acceptors],
        ],
        dtype=object,
    )

    # for i in range(len(name_uni.atoms)):
    #     print(name_uni.atoms[i], traj_uni.atoms[i])

    # print(stacked_array[np.where(stacked_array[:, 1] == 33)])
    # for i in stacked_array[:, 1:4]:
    #     print(traj_uni.atoms[int(i[0])], traj_uni.atoms[int(i[2])])

    hb_matrix = np.zeros([len(donors), len(acceptors)])
    for i, d in enumerate(donors):
        print(d.ix, d.ix)
        for j, a in enumerate(acceptors):
            print(a.ix, a.ix)
            match = np.where(
                (stacked_array[:, 1] == d.ix) & (stacked_array[:, 3] == a.ix)
            )[0]
            if match.size:
                hb_matrix[i, j] = len(match)

    print(resids)
    sorted_donors = sort_list(resids[0], str_reverse=False)
    sorted_acceptors = sort_list(resids[1], str_reverse=True)
    print(sorted_donors)
    print(sorted_acceptors)

    order_donors = [list(resids[0]).index(d) for d in sorted_donors]
    order_acceptors = [list(resids[1]).index(d) for d in sorted_acceptors]

    hb_matrix = hb_matrix[:, order_acceptors]
    hb_matrix = hb_matrix[order_donors, :]

    print(hb_matrix)

    hb_solvent_donors_array = np.zeros([len(donors)])
    hb_solvent_acceptors_array = np.zeros([len(acceptors)])
    acceptor_ix = np.array([a.ix for a in acceptors.atoms])
    donors_ix = np.array([a.ix for a in donors.atoms])
    for i, d in enumerate(donors):
        match = np.where(
            (stacked_array[:, 1] == d.ix)
            & np.isin(stacked_array[:, 3], acceptor_ix, invert=True)
        )[0]
        if match.size:
            hb_solvent_donors_array[i] = len(match)
    for j, a in enumerate(acceptors):
        match = np.where(
            (stacked_array[:, 3] == a.ix)
            & np.isin(stacked_array[:, 1], donors_ix, invert=True)
        )[0]
        if match.size:
            hb_solvent_acceptors_array[j] = len(match)
    print(hb_solvent_donors_array)
    print(hb_solvent_acceptors_array)

    # if output_prefix:
    #     np.save(f"{output_prefix}_avg.npy", avg_matrix)
    #     np.save(f"{output_prefix}_all_matrix.npy", stacked_matrix)
    #     np.save(f"{output_prefix}_all_array.npy", stacked_array)
    #     np.save(f"{output_prefix}_indexes.npy", resids)
    #     print(f"Data has been saved.")

    print(f"[DONE] {traj_trr} in {time() - start_time:.2f}\n")

    return avg_matrix, stacked_matrix, resids


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print(f"No files found in {INPUT_DIR} matching {GLOB}.")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")

    avg_matrix, all_matrices, resids = universe_setup(
        files, NAMED_PDB, GEOM_TPR, output_prefix=SAVE_PREFIX
    )

    print(resids)
    print(avg_matrix)
