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


def plot_histogram(data, title, bins=50):
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
    glob_pattern = f"boc-pg*/sasa_{title}.csv"

    files = sorted(base_dir.glob(glob_pattern))
    order = ["pgs4", "pgsssr", "pgsrss", "pgsr4"]

    # sortujemy wg pozycji w liście 'order'
    files_sorted = sorted(
        files,
        key=lambda f: order.index(
            f.parts[0].split("-")[1]
        ),  # wyciągamy np. 'pgs4' z 'boc-pgs4'
    )

    print(files_sorted)

    system_names = []
    for path in files_sorted:
        system_names.append(name_map[os.path.basename(os.path.dirname(path))])

    sasa_statistics = open(f"sasa-{title}.dat", "w")
    sasa_statistics.write("system_name\tmean_sasa\tstd_sasa\n")

    sasa_list = []

    for file in files_sorted:
        match = re.search(r"(boc-.+)/", file.as_posix())
        if not match:
            print(f" Could not extract key from {file}")
            continue

        system_key = match[1]
        system_name = name_map[system_key]

        data = pd.read_csv(file)

        mean_sasa = np.mean(data["0"])
        std_sasa = np.std(data["0"])

        sasa_list.append([mean_sasa, std_sasa])

        sasa_statistics.write(
            f"{system_name.replace(' ', '_')}\t{mean_sasa}\t{std_sasa}\n"
        )

        # os.chdir(system_key)
        # plot_ma(data["0"], title)
        # plot_histogram(data["0"], title)
        # os.chdir("..")
    sasa_statistics.close()

    df = pd.DataFrame(np.array(sasa_list), columns=["Mean", "Std"], index=system_names)

    plt.figure(figsize=(6, 3))
    plt.bar(system_names, df["Mean"], yerr=df["Std"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("SASA (Å^2)")
    plt.title(f"SASA per System, selection: {title}")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"SASA_{title}_bar.png", dpi=600)
