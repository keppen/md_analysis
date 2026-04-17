import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd


def load_xvg(filename):
    with open(filename) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if line.startswith(("#", "@")):
            continue
        data.append(list(map(float, line.strip().split())))
    return np.array(data)


# Energy component names
legends = [
    "Electrostatic - Short Range",
    "Dispersive - Short Range",
    "Electrostatic - 1-4 Bonded",
    "Dispersive - 1-4 Bonded",
]

name_map = {
    "boc-a4": "Boc-A4",
    "boc-ds4": "Boc-D4 (SSSS)",
    "boc-dsr4": "Boc-D4 (SRSR) ",
    "boc-cs4": "Boc-C4 (SSSS)",
    "boc-csr4": "Boc-C4 (SRSR)",
    "boc-css4": "Boc-Cc4 (SS-SS-SS-SS)",
    "boc-cssrr4": "Boc-Cc4 (SS-RR-SS-RR)",
    "boc-vs4": "Boc-V4 (SSSS)",
    "boc-vsr4": "Boc-V4 (SRSR)",
    "boc-ls4": "Boc-L4 (SSSS)",
    "boc-lsr4": "Boc-L4 (SRSR)",
    "boc-pas4": "Boc-Pa4 (SSSS)",
    "boc-pasr4": "Boc-Pa4 (SRSR)",
    "boc-pgs4": "Boc-Pg4 (SSSS)",
    "boc-pgsrss": "Boc-Pg4 (SRSS)",
    "boc-pgsssr": "Boc-Pg4 (SSSR)",
    "boc-pgsr4": "Boc-Pg4 (SRSR)",
    "boc-pss4": "Boc-Pc4 (SS-SS-SS-SS)",
    "boc-pssrr4": "Boc-Pc4 (SS-RR-SS-RR)",
}

# Load all relevant energy files
# Paths to cluster data files. Code sets '.' as cwd.
# E.g. boc-pg* implies that ./boc-pg*/ener.xvg file exist.
all_paths = sorted(glob.glob("boc-pg*/ener.xvg"))
avg_energies = []
system_names = []

for path in all_paths:
    data = load_xvg(path)
    averages = np.mean(data[:, 1:], axis=0)
    avg_energies.append(averages)
    system_names.append(name_map[os.path.basename(os.path.dirname(path))])

# Create DataFrame
df = pd.DataFrame(avg_energies, columns=legends, index=system_names)

# Define colors
color_map = {
    "Electrostatic - Short Range": "#1F77B4",  # blue
    "Dispersive - Short Range": "#D62728",  # red
    "Electrostatic - 1-4 Bonded": "#17BECF",  # cyan-blue
    "Dispersive - 1-4 Bonded": "#FF7F0E",  # orange-red
}


# Plot each energy component separately
for energy_type in legends:
    plt.figure(figsize=(12, 5))
    values = df[energy_type]
    plt.bar(system_names, values, color=color_map[energy_type])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Energy (kJ/mol)")
    plt.title(f"{energy_type} Energy per System")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{energy_type.replace(' ', '_')}_energy_bar.png", dpi=600)
    # plt.show()
