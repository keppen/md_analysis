import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path

# ---- INPUT ----
PREFIX = sys.argv[1]  # e.g. "centroids"
# ----------------


def plot_all_pair_histograms(all_mats, resids, outdir="histograms"):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    n_res = len(resids)

    # --- Font: Arial-like on Linux ---
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Liberation Sans", "DejaVu Sans"]
    mpl.rcParams["font.size"] = 7.5

    # --- Line widths ---
    mpl.rcParams["lines.linewidth"] = 1
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["ytick.major.width"] = 0.5
    cm_to_inch = 1 / 2.54
    fig_size = 6 * cm_to_inch  # square

    for i in range(n_res):
        for j in range(i + 1, n_res):
            values = all_mats[:, i, j]

            fig, ax = plt.subplots(figsize=(fig_size, fig_size / 1.5))

            ax.hist(values, bins=50)

            ax.set_xlabel("Maximum VdW overlap (Å)")
            ax.set_ylabel("Count")

            ax.set_title(f"Residues {resids[i]}-{resids[j]}")

            ax.axvline(0.0, linestyle="--", linewidth=0.8)

            plt.tight_layout()

            plt.savefig(
                outdir / f"overlap_{resids[i]}_{resids[j]}.pdf",
                dpi=300,
            )
            plt.close(fig)


def load_data(prefix):
    avg = np.load(f"{prefix}_overlap_all_matrices.npy")
    resids = np.loadtxt(f"{prefix}_overlap_resids.dat.dat", dtype=int)
    return all_mats, resids


if __name__ == "__main__":
    avg, all_mats, resids = load_data(PREFIX)

    # plot histogram of each pair of matrix

    print("[DONE] Plots saved.")
