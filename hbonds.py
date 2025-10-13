from os import chdir, listdir
from sys import argv
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
import parmed as pmd
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
import multiprocessing as mp

np.set_printoptions(threshold=np.inf, linewidth=200)


def trr_filter(file: str) -> bool:
    if file.split(".")[1] == "trr":
        return True
    return False


def get_vector(p0, p1):
    return p1 - p0


def get_angle(v0, v1):
    len0 = np.linalg.norm(v0)
    len1 = np.linalg.norm(v1)
    return np.dot(v0, v1) / len0 / len1


root: str = argv[1]
# topology_file: str = argv[2]
name: str = argv[2]
trr_files = list(filter(trr_filter, listdir(root)))

chdir(root)

topology_gro = f"../initialize/{name}.top"
coordinates_gro = f"10ns_{name}-1.gro"

top_obj = pmd.gromacs.GromacsTopologyFile(topology_gro)
# coord_obj = pmd.gromacs.GromacsGroFile.parse(coordinates_gro)

print(top_obj.residues[1])
print(top_obj.residues[1].atoms)
print(top_obj.bonds)


# top_obj.positions = coord_obj.positions

top_obj.save("system.top", overwrite=True)

# exit()

print(trr_files)

universe = mda.Universe("system.top", trr_files[:200], topology_format="ITP")

donor_selection = "resname UNK and name N N1 N2 N3 N4"
# hydrogen_selection = "name H6 H12 H18 H24 H31"
acceptor_selection = "resname UNK and name O1 O3 O5 O7"

donors = universe.select_atoms(donor_selection)
acceptors = universe.select_atoms(acceptor_selection)

print(donors.indices)
print(acceptors.indices)
print(universe.select_atoms("index 2"))
print(universe.select_atoms("resname UNK or around 5 resname UNK"))

# exit()


def process_chunk(args):
    """Process a chunk of frames with optimized trajectory handling"""
    topology, trajectory, config, chunk = args
    universe = mda.Universe(topology, trajectory, topology_format="ITP")

    # Create analysis instance with pre-computed selections
    hbonds = HydrogenBondAnalysis(
        universe=universe,
        donors_sel=config["donors_sel"],
        hydrogens_sel=config["hydrogens_sel"],
        acceptors_sel=config["acceptors_sel"],
        d_a_cutoff=config["d_a_cutoff"],
        d_h_a_angle_cutoff=config["d_h_a_angle_cutoff"],
        update_selections=False,
    )

    # Process frames using MDAnalysis' efficient trajectory slicing
    hbonds.run(start=chunk[0], stop=chunk[1], step=1, verbose=True)
    return hbonds.results.hbonds


# Original analysis setup
hbonds = HydrogenBondAnalysis(
    universe=universe,
    d_a_cutoff=3.0,
    d_h_a_angle_cutoff=150,
    update_selections=False,
)

# Pre-compute selections in main process first
update_selections = True
unk_hydrogens_sel = hbonds.guess_hydrogens("resname UNK")
sol_hydrogens_sel = hbonds.guess_hydrogens("resname SOL")
unk_acceptors_sel = hbonds.guess_acceptors("resname UNK")
sol_acceptors_sel = hbonds.guess_acceptors("resname SOL")

hydrogens_sel = f"({unk_hydrogens_sel})"

if sol_hydrogens_sel:
    hydrogens_sel += (
        f" or ({sol_hydrogens_sel} and around 3.5 resname UNK)"  # Fixed syntax
    )
acceptors_sel = f"({unk_acceptors_sel})"

if sol_acceptors_sel:
    acceptors_sel += (
        f" or ({sol_acceptors_sel} and around 3.5 resname UNK)"  # Fixed syntax
    )
if not sol_hydrogens_sel and not sol_acceptors_sel:
    update_selections = False

# Store final selections in config
config = {
    "donors_sel": None,
    "hydrogens_sel": hydrogens_sel,
    "acceptors_sel": acceptors_sel,
    "d_a_cutoff": 3.0,
    "d_h_a_angle_cutoff": 150,
    "update_selections:": update_selections,
}

# Create optimized chunks using frame ranges
n_frames = len(universe.trajectory)
n_workers = 10
chunk_size = n_frames // n_workers  # Smaller dynamic chunks
chunks = [(i, min(i + chunk_size, n_frames)) for i in range(0, n_frames, chunk_size)]
print(chunks)

# Prepare arguments with pre-computed config
args = [("system.top", trr_files[:200], config, chunk) for chunk in chunks]

# Use imap_unordered for dynamic load balancing
with mp.Pool(processes=n_workers) as pool:
    results = []
    for result in pool.imap_unordered(process_chunk, args, chunksize=2):
        results.append(result)

# Combine results from all workers
combined_hbonds = np.concatenate(results)

# Update the original analysis object with combined results
hbonds.results.hbonds = combined_hbonds

result = hbonds  # Maintain original variable name
hb_array = result.count_by_ids()

# print(hb_array)

np.save("hbond", result.hbonds)

unique_donors = np.unique(result.count_by_ids()[:, 0])

unk_res = universe.residues[0].atoms.indices
unk_res = [i + 1 for i in unk_res]

unique_donors = [i for i in unique_donors if i in unk_res]

r = np.array(
    [
        [val] + list(hb_array[hb_array[:, 0] == val, 1:].sum(axis=0))
        for val in unique_donors
    ]
)

np.save("amines_all", r[:, [0, 3]])
print(r[:, [0, 3]])

hb_matrix = np.zeros([len(donors), len(acceptors)])

for i, d in enumerate(donors):
    for j, a in enumerate(acceptors):
        matches = np.where((hb_array[:, 0] == d.ix + 1) & (hb_array[:, 2] == a.ix + 1))[
            0
        ]  # Extract indices

        if matches.size > 0:  # Ensure there's at least one match
            hb_matrix[i, j] = hb_array[
                matches, 3
            ].sum()  # Sum the values if multiple matches exist
print(hb_matrix)

np.save("hb-matrix", hb_matrix)


r = np.array(
    [
        [val]
        + list(
            hb_array[
                (hb_array[:, 0] == val) & np.isin(hb_array[:, 2], unk_res), 1:
            ].sum(axis=0)
        )
        for val in unique_donors
    ]
)

print(r[:, [0, 3]])
