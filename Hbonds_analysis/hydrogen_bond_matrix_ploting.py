import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
    "boc-pgsssr": "Boc-Pg4 (SSSR), CHCl3",
    "boc-pgsrss": "Boc-Pg4 (SRSS), CHCl3",
    "boc-pgsr4": "Boc-Pg4 (SRSR), CHCl3",
    "boc-pss4": "Boc-Pc4 (SS-SS-SS-SS)",
    "boc-pssrr4": "Boc-Pc4 (SS-RR-SS-RR)",
}
name_map_acn = {
    "boc-pgs4": "Boc-Pg4 (SSSS), ACN",
    "boc-pgsssr": "Boc-Pg4 (SSSR), ACN",
    "boc-pgsrss": "Boc-Pg4 (SRSS), ACN",
    "boc-pgsr4": "Boc-Pg4 (SRSR), ACN",
}


def sort_list(lst, str_reverse=False):
    import random

    def get_num(item):
        return int(re.search(r"-(\d+):", item).group(1))

    def get_str(item):
        return re.search(r":([A-Za-z]+)", item).group(1)

    lst = sorted(lst, key=get_num)  # num rosnąco
    lst = sorted(lst, key=get_str, reverse=str_reverse)  # str malejąco
    return lst


def plot_heatmap(
    matrix: np.ndarray,
    title: str = "Heatmap",
    xlabels: Optional[Sequence[str]] = None,
    ylabels: Optional[Sequence[str]] = None,
    cmap: str = "viridis",
    annotate: bool = False,
    fmt: str = ".2f",
    figsize: tuple[float, float] = (7.0, 6.0),
    show_colorbar: bool = True,
    normalized=False,
):
    """
    Plot a 2D heatmap using pure Matplotlib.

    Parameters
    ----------
    matrix : np.ndarray
        2D numeric array.
    title : str
        Plot title.
    xlabels : list of str, optional
        Labels for columns (x-axis).
    ylabels : list of str, optional
        Labels for rows (y-axis).
    cmap : str
        Colormap name.
    annotate : bool
        If True, print values inside cells.
    fmt : str
        Format for annotations.
    figsize : tuple
        Figure size.
    show_colorbar : bool
        Whether to include a colorbar.
    """

    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    matplotlib.rcParams.update({"font.size": 8})

    fig, ax = plt.subplots(figsize=figsize)
    print("figsize ", figsize)

    # Draw heatmap
    if normalized:
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="upper", vmin=0, vmax=1)
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="upper")

    # Add colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(title, rotation=-90, va="bottom")

    # Configure ticks and labels
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticks(np.arange(matrix.shape[1]))
    print(matrix.shape, xlabels, ylabels)
    if ylabels is not None:
        ax.set_yticklabels(ylabels, rotation=45, ha="right")
    if xlabels is not None:
        ax.set_xticklabels(xlabels, rotation=45, ha="right")

    # Optional annotations
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = format(matrix[i, j], fmt)
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    # Titles and layout
    # ax.set_title(title)
    fig.tight_layout()
    plt.savefig(f"{title.replace(' ', '-')}-hbond-matrix.png", dpi=1200)
    plt.close()


if __name__ == "__main__":
    base_dir = Path(".")
    currrent_cwd = os.getcwd()

    glob_indexes = f"hbonds_indexes.npy"

    glob_avg_chcl3 = f"analysis/boc-*/hbonds_avg.npy"

    files_chcl3 = sorted(base_dir.glob(glob_avg_chcl3))

    glob_avg_acn = f"analysis-acn/boc-*/hbonds_avg.npy"

    files_acn = sorted(base_dir.glob(glob_avg_acn))

    files_chcl3_sorted = files_chcl3
    files_acn_sorted = files_acn

    files_sorted = []
    files_sorted.extend(files_chcl3_sorted)
    files_sorted.extend(files_acn_sorted)

    system_names = []
    matrixes = []
    for path in files_chcl3_sorted:
        system_names.append(name_map_chcl3[os.path.basename(os.path.dirname(path))])

    for path in files_acn_sorted:
        system_names.append(name_map_acn[os.path.basename(os.path.dirname(path))])

    for file in files_sorted:
        match = re.search(r"(boc-.+)/", file.as_posix())
        if not match:
            print(f" Could not extract key from {file}")
            continue

        system_key = match[1]
        system_name = name_map[system_key]

        data = np.load(file)
        matrixes.append(data)
        print(file.parent)
        print(f"{system_key}/{glob_indexes}")
        print(data)
        indexes = np.load(f"{file.parent}/{glob_indexes}", allow_pickle=True)
        donors = np.array(indexes[0])
        acceptors = np.array(indexes[1])

        sorted_donors = sort_list(donors, str_reverse=False)
        sorted_acceptors = sort_list(acceptors, str_reverse=True)

        order_acceptors = [list(acceptors).index(d) for d in sorted_acceptors]

        data = data[:, order_acceptors]
        print(order_acceptors)
        print(data)
        print(sorted_donors, sorted_acceptors)

        os.chdir(file.parent)
        plot_heatmap(
            data,
            title=f"{system_name.replace(' ', '-')}-Average-Hydrogen-Bond",
            ylabels=sorted_donors,
            xlabels=sorted_acceptors,
            cmap="BuPu",
            annotate=False,
            figsize=(3.8, 3.25),
        )
        os.chdir(currrent_cwd)

    flattened = [m.flatten() for m in matrixes]
    similarity_matrix = cosine_similarity(flattened)

    # selection = [n for n in system_names if "SSSS" in n or "A4" in n]
    # idx = np.array([i for i, n in enumerate(system_names) if n in selection])
    # print(similarity_matrix[np.ix_(idx, idx)].shape)
    # plot_heatmap(
    #     similarity_matrix[np.ix_(idx, idx)],
    #     title="Sel-SSSS-Similarity",
    #     xlabels=selection,
    #     ylabels=selection,
    #     cmap="RdPu",
    # )
    #
    # selection = [n for n in system_names if "SRSR" in n or "A4" in n]
    # idx = np.array([i for i, n in enumerate(system_names) if n in selection])
    # print(similarity_matrix[np.ix_(idx, idx)].shape)
    # plot_heatmap(
    #     similarity_matrix[np.ix_(idx, idx)],
    #     title="Sel-SRSR-Similarity",
    #     xlabels=selection,
    #     ylabels=selection,
    #     cmap="RdPu",
    # )
    #
    # keywords = ["Pc4", "A4", "Cc4"]
    # selection = [n for n in system_names if any(k in n for k in keywords)]
    # idx = np.array([i for i, n in enumerate(system_names) if n in selection])
    # print(keywords, idx)
    # print(similarity_matrix[np.ix_(idx, idx)].shape)
    # plot_heatmap(
    #     similarity_matrix[np.ix_(idx, idx)],
    #     title="Sel-Cyclic-Similarity",
    #     xlabels=selection,
    #     ylabels=selection,
    #     cmap="RdPu",
    # )
    #
    # keywords = ["A4", "D4", "V4", "L4", "Pg4", "Pa4"]
    # keyword2 = "SSSS"
    # selection = []
    # for i, k in enumerate(keywords):
    #     for j, n in enumerate(system_names):
    #         if k in n and keyword2 in n:
    #             selection.append(n)
    #         elif k == "A4" and k in n:
    #             selection.append(n)
    #         else:
    #             continue
    # idx = np.array([system_names.index(n) for n in selection])
    # print(system_names)
    # print(keywords, idx)
    # print(similarity_matrix[np.ix_(idx, idx)].shape)
    # plot_heatmap(
    #     similarity_matrix[np.ix_(idx, idx)],
    #     title="Cosine similarity between HBond matrices",
    #     # xlabels=selection,
    #     # ylabels=selection,
    #     cmap="Reds",
    #     figsize=(3.8, 3.25),
    # )
    #

    order = ["(SSSS),", "(SSSR),", "(SRSS),", "(SRSR),"]

    def selection_sort(selection, order):
        selection_sorted = sorted(
            selection,
            key=lambda s: order.index(
                s.split()[1]
            ),  # wyciągamy np. 'pgs4' z 'boc-pgs4'
        )
        return selection_sorted

    keywords = ["Pg4"]
    selection = [n for n in system_names if any(k in n for k in keywords)]
    keywords = ["CHCl3"]
    selection = [n for n in selection if any(k in n for k in keywords)]
    selection = selection_sort(selection, order)
    idx = np.array([i for i, n in enumerate(system_names) if n in selection])
    print(keywords, idx)
    print(similarity_matrix[np.ix_(idx, idx)].shape)
    plot_heatmap(
        similarity_matrix[np.ix_(idx, idx)],
        title="PG4-CHCL3-Similarity",
        xlabels=selection,
        ylabels=selection,
        cmap="RdPu",
        normalized=True,
    )

    keywords = ["Pg4"]
    selection = [n for n in system_names if any(k in n for k in keywords)]
    keywords = ["ACN"]
    selection = [n for n in selection if any(k in n for k in keywords)]
    selection = selection_sort(selection, order)
    idx = np.array([i for i, n in enumerate(system_names) if n in selection])
    print(keywords, idx)
    print(similarity_matrix[np.ix_(idx, idx)].shape)
    plot_heatmap(
        similarity_matrix[np.ix_(idx, idx)],
        title="PG-ACN-Similarity",
        xlabels=selection,
        ylabels=selection,
        cmap="RdPu",
        normalized=True,
    )
    keywords = ["Pg4"]
    selection = [n for n in system_names if any(k in n for k in keywords)]
    keywords1 = ["ACN"]
    selection1 = [n for n in selection if any(k in n for k in keywords1)]
    selection1 = selection_sort(selection1, order)
    keywords2 = ["CHCl3"]
    selection2 = [n for n in selection if any(k in n for k in keywords2)]
    selection2 = selection_sort(selection2, order)
    idx1 = np.array([i for i, n in enumerate(system_names) if n in selection1])
    idx2 = np.array([i for i, n in enumerate(system_names) if n in selection2])
    print(keywords, idx1)
    print(keywords, idx2)
    print(similarity_matrix[np.ix_(idx1, idx2)].shape)
    plot_heatmap(
        similarity_matrix[np.ix_(idx1, idx2)],
        title="PG-ACNvsCHCL3-Similarity",
        xlabels=selection2,
        ylabels=selection1,
        cmap="RdPu",
        normalized=True,
    )
