import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys

# ---- SETTINGS ----
DATA_DIR = Path(sys.argv[1])
SUFFIX = sys.argv[2]
POLY_ATOM = sys.argv[3]
SOLV_ATOM = sys.argv[4]
OUTPUT = SUFFIX + "_rdf_plot.png"
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
    "#FB6A4A",
    "#FC9272",
    "#FCBBA1",
    "#A50F15",
    "#67000D",
    "#CB181D",
    "#E31A1C",
    "#EF3B2C",
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

new_colors = [
    "#08306B",
    "#7F0000",
    "#333333",
    "#8B8000",
    "#3F007D",
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
        solv_atom = parts[-4]
        resname = parts[-3]
        resid = parts[-2]
        atom = parts[-1]
        return f"{resname}-{resid} {atom}:SOLV-{solv_atom}"
    except:
        return name


def main():
    bins_file = DATA_DIR / "rdf_bins.npy"

    if not bins_file.exists():
        print("[ERROR] rdf_bins.npy not found")
        sys.exit(1)

    bins = np.load(bins_file)
    rdf_files = []
    # rdf_files.extend(sorted(DATA_DIR.glob(f"rdf_{SUFFIX}*SOLVENT_C_*OT.npy")))
    # rdf_files.extend(sorted(DATA_DIR.glob(f"rdf_{SUFFIX}*SOLVENT_C_*O.npy")))

    print(f"rdf_{SUFFIX}*SOLVENT_{SOLV_ATOM}_*_{POLY_ATOM}*.npy")

    rdf_files.extend(
        sorted(DATA_DIR.glob(f"rdf_{SUFFIX}*SOLVENT_{SOLV_ATOM}_*_{POLY_ATOM}*.npy"))
    )
    if POLY_ATOM == "OT":
        rdf_files.extend(
            sorted(DATA_DIR.glob(f"rdf_{SUFFIX}*SOLVENT_{SOLV_ATOM}_*_O.npy"))
        )
        rdf_files.extend(
            sorted(DATA_DIR.glob(f"rdf_{SUFFIX}*SOLVENT_{SOLV_ATOM}_*_3_OA.npy"))
        )
    if POLY_ATOM == "HN":
        rdf_files.extend(
            sorted(DATA_DIR.glob(f"rdf_{SUFFIX}*SOLVENT_{SOLV_ATOM}_*_HO.npy"))
        )

    # rdf_files.extend(sorted(DATA_DIR.glob(f"rdf_{SUFFIX}*SOLVENT_H_*OT.npy")))
    # rdf_files.extend(sorted(DATA_DIR.glob(f"rdf_DONOR*.npy")))

    if not rdf_files:
        print("[ERROR] No rdf_*.npy files found")
        sys.exit(1)

    print(f"Found {len(rdf_files)} RDF files")

    plt.figure(figsize=(8, 6))

    all_rdfs = []

    for i, f in enumerate(rdf_files):
        rdf = np.load(f)
        label = extract_label(f)
        print(label.split(":")[0].split(" ")[1])

        if "O" in label.split(":")[0].split(" ")[1]:
            if "A" in label.split(":")[0].split(" ")[1]:
                color = blue_colors[i % 10]
            else:
                color = red_colors[i % 10]
        elif "N" in label.split(":")[0].split("-")[1]:
            color = blue_colors[i % 10]
        elif "H" in label.split(":")[0].split("-")[1]:
            color = grey_colors[i % 10]
        elif "COM" in label.split(":")[-1]:
            color = grey_colors[i % 10]
        else:
            color = grey_colors[i % 10]

        color = new_colors[i % 10]
        if i < MAX_CURVES:
            plt.plot(bins, rdf, alpha=0.9, label=label, color=color)

        all_rdfs.append(rdf)

    # --- average curve ---
    avg_rdf = np.mean(all_rdfs, axis=0)
    plt.plot(bins, avg_rdf, color="black", linewidth=2.5, label="Average")

    # --- plot styling ---
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title("RDF")

    if len(rdf_files) <= MAX_CURVES:
        plt.legend(fontsize=8)

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=300)

    print(f"[DONE] Saved plot → {OUTPUT}")

    plt.show()


if __name__ == "__main__":
    main()
