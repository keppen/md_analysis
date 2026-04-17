import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from matplotlib import cm
from matplotlib.patches import Patch
import re
import os

name_map = {
    "boc-a4": "Boc-A4",
    "boc-a4-empty": "Boc-A4",
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


def parse_cluster_data(file_content):
    """Parse the cluster data from file content into cluster IDs and counts."""
    clusters, sizes = [], []
    for line in file_content.splitlines():
        if not line or line.startswith(("@", "#", "gmx")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                clusters.append(int(parts[0]))
                sizes.append(int(parts[1]))
            except ValueError:
                continue
    return np.array(clusters), np.array(sizes)


def load_dataset(path):
    """Load clusters and percent sizes from a .txt file"""
    with open(path) as f:
        clusters, sizes = parse_cluster_data(f.read())
    total = sizes.sum()
    percentages = (sizes / total) * 100
    return clusters, sizes


def plot_stacked_bars(datasets, labels, title_template, output_file_template):
    """
    Plot separate stacked bar charts for each dataset based on the cluster distributions.
    """

    # Initialize an empty list to store the DataFrames for each dataset
    n_datasets = len(datasets)
    threshold = 0.85

    # Initialize an empty DataFrame with columns corresponding to the labels
    dfs = pd.DataFrame(columns=labels)
    other_dict = {}
    num_other_dict = {}

    # Loop through each dataset
    for i, dataset in enumerate(datasets):
        print(dataset)
        # Ensure labels[i] is a list of column names
        column_names = [labels[i]]

        # Convert the dataset into a pandas DataFrame, transpose if necessary
        df = pd.DataFrame(dataset[1].T, columns=column_names)

        total = df[labels[i]].sum()
        cumsum = df[labels[i]].cumsum()

        top_rows = df[cumsum < total * threshold]
        if top_rows.empty:
            top_rows = df.iloc[:1]

        # Reset index for alignment
        top_col = top_rows.reset_index(drop=True)

        # Pad to current max length in dfs
        max_len = max(len(top_col), len(dfs))  # get new total row count
        top_col = top_col.reindex(range(max_len), fill_value=0)
        dfs = dfs.reindex(range(max_len))  # extend dfs if needed

        print(top_rows)

        remaining_rows = df.drop(top_rows.index)
        num_merged = len(remaining_rows)
        num_other_dict[labels[i]] = num_merged

        other_sum = remaining_rows.sum()
        other_dict[labels[i]] = other_sum

        # Assign the padded DataFrame to the appropriate column in dfs
        dfs[labels[i]] = top_col

    dfs = dfs / 1000
    for k, v in other_dict.items():
        other_dict[k] = v / 1000
    print(dfs)
    print(dfs.sum())

    matplotlib.rcParams.update({"font.size": 8})

    # Now loop through the DataFrames and plot the stacked bar charts
    fig, ax = plt.subplots(figsize=(6.3, 3.4))

    # Red gradient colormap
    cmap = cm.Reds  # You can change this to other colormaps if needed
    norm = plt.Normalize(vmin=0, vmax=len(dfs))

    # Plot the stacked bar chart
    bottom = np.zeros(len(labels))
    legend_patches = []  # To store legend patches

    for i, row in enumerate(dfs.iterrows()):
        array_row = row[1].to_numpy()

        # Get color from the colormap based on the row index
        color = cmap(norm(i))

        # Plot the bars
        ax.bar(
            range(len(labels)),
            array_row,
            bottom=bottom,
            width=0.5,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += np.nan_to_num(array_row, nan=0.0)
        # Create a legend entry for the current cluster (index i)
        legend_patches.append(Patch(color=color, label=f"C{i + 1}"))

    color = "grey"
    ax.bar(
        range(len(labels)),
        [y.iloc[0] for _, y in other_dict.items()],
        bottom=bottom,
        width=0.5,
        color=color,
        edgecolor="black",
        linewidth=0.5,
    )
    legend_patches.append(Patch(color=color, label="Others"))
    for x, label in enumerate(labels):
        value = float(other_dict[label].iloc[0])
        count = num_other_dict[label]

        # Calculate vertical position: center of the "Other" bar
        # y_pos = bottom[x] + value / 2
        y_pos = 92

        print(x, y_pos)

        ax.text(
            x,
            y_pos,
            f"{count}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if value > 5 else "black",  # adjust text color for visibility
            fontweight="bold",
        )

    # Set axis labels and title
    plt.xticks(range(len(labels)), [name_map[l] for l in labels], rotation=45)
    ax.set_ylim([0, 100])
    ax.set_ylabel("Population [%]")
    # ax.set_title(f"{title_template}")
    ax.grid(axis="y", linestyle="--", alpha=0.2)

    # Add the legend to the plot
    # ax.legend(handles=legend_patches, title="Clusters", bbox_to_anchor=[1.12, 1])

    # Save the plot with a unique filename based on the dataset's label
    output_file = f"cluster-stacked.png"
    plt.tight_layout()
    fig.savefig(output_file, dpi=1200)
    plt.close(fig)  # Close the figure to prevent display


def main():

    def extract_stereo_from_label(label):
        match = re.search(r"\((SSSS|SSSR|SRSS|SRSR)\)", label)
        if match:
            return match.group(1)
        return None  # for things like Boc-A4

    def sort_key(path):
        dirname = os.path.basename(path)

        if dirname not in name_map:
            return (1, dirname)  # push unknowns to end

        label = name_map[dirname]
        stereo = extract_stereo_from_label(label)

        if stereo is None:
            return (1, label)  # non-stereochemical systems at end

        return (0, order.index(stereo))

    order = ["SSSS", "SSSR", "SRSS", "SRSR"]

    parser = argparse.ArgumentParser(
        description="Plot separate stacked bar charts of cluster distributions for multiple datasets."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to cluster data files. Code sets '.' as cwd. E.g. boc-pg* implies that ./boc-pg*/RMSD/ direcotry exist.",
    )
    parser.add_argument(
        "--output",
        default="cluster_stacked_{label}.png",
        help="Output plot filename template",
    )
    args = parser.parse_args()

    # if args.labels and len(args.labels) != len(args.files):
    #     parser.error("Number of labels must match number of files")

    args.files = sorted(args.files, key=sort_key)

    data_file = [
        f"{f}/RMSD/{f}-size.xvg" for f in args.files
    ]  # change analysis directory structure here.
    print(data_file)

    labels = (
        args.files if args.files else [f"Data {i + 1}" for i in range(len(args.files))]
    )
    datasets = [load_dataset(fp) for fp in data_file]

    plot_stacked_bars(
        datasets,
        labels,
        "Cluster Size Distribution",
        args.output,
    )


if __name__ == "__main__":
    main()
