import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ---- INPUT ----
PREFIX = sys.argv[1]  # e.g. "centroids"
# ----------------


def load_data(prefix):
    avg = np.load(f"{prefix}_avg_matrices.npy")
    all_mats = np.load(f"{prefix}_dist_matrices.npy")
    resids = np.loadtxt(f"{prefix}_resids.dat", dtype=int)
    return avg, all_mats, resids


def plot_heatmap(matrix, resids, title, outfile, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")

    # optional: label ticks sparsely
    step = max(1, len(resids) // 20)
    ticks = np.arange(0, len(resids), step)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(resids[ticks], rotation=90)
    ax.set_yticklabels(resids[ticks])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Distance (Å)")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_variance(all_mats, resids, outfile):
    std_mat = np.std(all_mats, axis=0)

    plot_heatmap(
        std_mat, resids, title="Distance fluctuation (std dev)", outfile=outfile
    )


def plot_contact_map(avg_mat, resids, cutoff, outfile):
    contact = avg_mat < cutoff

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(contact, origin="lower", aspect="auto")

    ax.set_title(f"Contact map (< {cutoff} Å)")
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")

    plt.colorbar(im, ax=ax, label="Contact (0/1)")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


if __name__ == "__main__":
    avg, all_mats, resids = load_data(PREFIX)

    # --- average distance matrix ---
    plot_heatmap(
        avg,
        resids,
        title="Average centroid distance",
        outfile=f"{PREFIX}_avg_heatmap.png",
    )

    # --- variance (dynamics insight) ---
    plot_variance(all_mats, resids, outfile=f"{PREFIX}_std_heatmap.png")

    # --- contact map ---
    plot_contact_map(avg, resids, cutoff=6.0, outfile=f"{PREFIX}_contact_map.png")

    print("[DONE] Plots saved.")
