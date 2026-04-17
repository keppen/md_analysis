import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys

# ---- SETTINGS ----
DATA_DIR = Path(sys.argv[1])
MODE = sys.argv[2]
OUTPUT = MODE + "_rdf_plot.png"
MAX_CURVES = 20  # avoid clutter
# ------------------

blue_colors = [
    "#08306B",
    "#08519C",
    "#2171B5",
    "#4292C6",
    "#6BAED6",
    "#9ECAE1",
    "#C6DBEF",
    "#DEEBF7",
    "#3182BD",
    "#1F4E79",
]

red_colors = [
    "#7F0000",
    "#B30000",
    "#CB181D",
    "#E31A1C",
    "#EF3B2C",
    "#FB6A4A",
    "#FC9272",
    "#FCBBA1",
    "#A50F15",
    "#67000D",
]

grey_colors = [
    "#1A1A1A",
    "#333333",
    "#4D4D4D",
    "#666666",
    "#808080",
    "#999999",
    "#B3B3B3",
    "#CCCCCC",
    "#E0E0E0",
    "#F2F2F2",
]

purple_colors = [
    "#3F007D",
    "#54278F",
    "#6A51A3",
    "#807DBA",
    "#9E9AC8",
    "#BCBDDC",
    "#DADAEB",
    "#EFEDF5",
    "#4A1486",
    "#7B3294",
]


def extract_label(filename):
    """
    Convert filename like:
    rdf_DONOR_N_UNK_12_N.npy
    -> UNK-12:N
    """
    name = filename.stem
    parts = name.split("_")

    try:
        solv_atom = parts[2]
        resname = parts[3]
        resid = parts[4]
        atom = parts[5]
        return f"UNK-{atom}:SOLV-{solv_atom}"
    except:
        return name


def main():
    bins_file = DATA_DIR / "rdf_bins.npy"

    if not bins_file.exists():
        print("[ERROR] rdf_bins.npy not found")
        sys.exit(1)

    bins = np.load(bins_file)

    rdf_files = sorted(DATA_DIR.glob(f"rdf_{MODE}*.npy"))

    if not rdf_files:
        print("[ERROR] No rdf_*.npy files found")
        sys.exit(1)

    print(f"Found {len(rdf_files)} RDF files")

    plt.figure(figsize=(8, 6))

    all_rdfs = []

    for i, f in enumerate(rdf_files):
        rdf = np.load(f)
        label = extract_label(f)

        if "O" in label.split(":")[0].split("-")[1]:
            color = red_colors[i % 10]
        elif "N" in label.split(":")[0].split("-")[1]:
            color = blue_colors[i % 10]
        elif "H" in label.split(":")[0].split("-")[1]:
            color = grey_colors[i % 10]
        else:
            color = False

        if i < MAX_CURVES:
            plt.plot(bins, rdf, alpha=0.9, label=label, color=color)

        all_rdfs.append(rdf)

    # --- average curve ---
    avg_rdf = np.mean(all_rdfs, axis=0)
    plt.plot(bins, avg_rdf, color="black", linewidth=2.5, label="Average")

    # --- plot styling ---
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title("RDF per urethane site")

    if len(rdf_files) <= MAX_CURVES:
        plt.legend(fontsize=8)

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=300)

    print(f"[DONE] Saved plot → {OUTPUT}")

    plt.show()


if __name__ == "__main__":
    main()
