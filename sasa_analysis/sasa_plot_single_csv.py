import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import re
import os

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

name_map_chcl3 = {
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
    "boc-pgs4": "Boc-Pg4 (SSSS), CHCl3",
    "boc-pgsrss": "Boc-Pg4 (SRSS), CHCl3",
    "boc-pgsssr": "Boc-Pg4 (SSSR), CHCl3",
    "boc-pgsr4": "Boc-Pg4 (SRSR), CHCl3",
    "boc-pss4": "Boc-Pc4 (SS-SS-SS-SS)",
    "boc-pssrr4": "Boc-Pc4 (SS-RR-SS-RR)",
}
name_map_acn = {
    "boc-pgs4": "Boc-Pg4 (SSSS), ACN",
    "boc-pgsrss": "Boc-Pg4 (SRSS), ACN",
    "boc-pgsssr": "Boc-Pg4 (SSSR), ACN",
    "boc-pgsr4": "Boc-Pg4 (SRSR), ACN",
}


def plot_ma(data, title):
    # plot SASA vs frame (with small moving average)
    n_points = np.arange(len(data))
    plt.figure(figsize=(8, 4))
    plt.plot(n_points, data, marker=".", linestyle="-", label="SASA per frame")
    window = len(n_points) // 100
    if len(data) >= window:
        ma = np.convolve(data, np.ones(window) / window, mode="same")
        plt.plot(n_points, ma, linewidth=2, label=f"{window}-frame moving avg")
    plt.xlabel("Frame")
    plt.ylabel("SASA (Å^2)")
    plt.title("SASA over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"sasa_over_time_{title}.png", dpi=1200)
    plt.close()
    print(f"Saved plot: sasa_over_time_{title}.png")


def plot_histogram(data, title, bins=100):
    """Plot a histogram of SASA values across frames."""
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("SASA (Å²)")
    plt.ylabel("Frequency")
    plt.title(f"SASA Distribution ({title})")
    plt.grid(axis="y", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"sasa_hist_{title}.png", dpi=1200)
    plt.close()
    print(f"Saved histogram: sasa_hist_{title}.png")


if __name__ == "__main__":
    title = sys.argv[1]
    base_dir = Path(".")
    glob_pattern = f"sasa_{title}.csv"
    #
    file = sorted(base_dir.glob(glob_pattern))[0]

    sasa_list = []

    data = pd.read_csv(file)

    mean_sasa = np.mean(data["0"])
    std_sasa = np.std(data["0"])

    sasa_list.append([mean_sasa, std_sasa])

    plot_ma(data["0"], title)
    plot_histogram(data["0"], title)

    df = pd.DataFrame(np.array(sasa_list), columns=["Mean", "Std"])
    print(df)
