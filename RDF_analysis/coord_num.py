import sys
import numpy as np
import MDAnalysis as mda

if len(sys.argv) != 4:
    print("Usage: script.py <rdf.npy> <cutoff>  <structure.gro>")
    sys.exit(1)

rdf_file = sys.argv[1]
rc = float(sys.argv[2])
solvent_sel = "resname LIG"
gro_file = sys.argv[3]

# --- load RDF ---
r = np.load("rdf_bins.npy")
g = np.load(rdf_file)

# --- compute density from structure ---
u = mda.Universe(gro_file)

solvent_atoms = u.select_atoms(solvent_sel)
solvent_residues = solvent_atoms.residues

N = len(solvent_residues)

box = u.dimensions[:3]
V = box[0] * box[1] * box[2]

rho = N / V  # molecules / A^3

# --- integration up to cutoff ---
mask = r <= rc
r_cut = r[mask]
g_cut = g[mask]

integrand = g_cut * r_cut**2
integral = np.trapz(integrand, r_cut)

coordination_number = 4 * np.pi * rho * integral

# --- output ---
print(f"Cutoff: {rc}")
print(f"Solvent selection: {solvent_sel}")
print(f"N (solvent molecules): {N}")
print(f"Box dims: {box} A^3")
print(f"Volume: {V:.3f} A^3")
print(f"Density: {rho:.6f} A^-3")
print(f"Coordination number: {coordination_number:.3f}")
