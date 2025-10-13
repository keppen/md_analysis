# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker


def configure_axes(ax, limits, labels, num_ticks=3, fontsize=9, pad=2):
    """
    Apply consistent styling to a 3D Axes object.
    """
    # Set limits and labels
    for dim, lim in enumerate(limits):
        axis_char = "xyz"[dim]
        getattr(ax, f"set_{axis_char}lim")(lim)
        getattr(ax, f"set_{axis_char}label")(
            labels[dim], fontsize=fontsize, labelpad=pad, rotation=[90, 90, 0][dim]
        )
    # Set pane transparency and axis lines
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0)
        axis.line.set_visible(True)
        axis.line.set_color("black")
        axis.line.set_linewidth(0.5)
        axis._axinfo["grid"].update(
            {"color": "grey", "linestyle": (0, (5, 5)), "linewidth": 0.5}
        )
    # Major ticks locator
    for dim, lim in enumerate(limits):
        base = (lim[1] - lim[0]) / num_ticks
        locator = ticker.MultipleLocator(base=base)
        axis_char = "xyz"[dim]
        getattr(ax, f"{axis_char}axis").set_major_locator(locator)
    # View angle
    ax.view_init(elev=25, azim=45)


def plot_kde_3d(
    density,
    coords,
    grid_limits,
    title,
    labels,
    downsample=1,
    alpha_power=0.9,
    threshold=1e-8,
    point_size=1,
    output_file=None,
):
    """
    3D scatter visualization of KDE density.
    """
    # Downsample
    data_ds = density[::downsample, ::downsample, ::downsample].astype(np.float32)
    coords_ds = coords[:, ::downsample, ::downsample, ::downsample].astype(np.float32)

    # Avoid zeros
    data_ds += threshold
    x, y, z = coords_ds
    x_flat, y_flat, z_flat = x.ravel(), y.ravel(), z.ravel()
    d_flat = data_ds.ravel()
    mask = d_flat > threshold
    x_flat, y_flat, z_flat, d_flat = (
        x_flat[mask],
        y_flat[mask],
        z_flat[mask],
        d_flat[mask],
    )

    # Color and alpha mapping
    d_min, d_max = d_flat.min(), d_flat.max()
    scaled = (d_flat - d_min) / (d_max - d_min)
    colors = plt.cm.inferno(scaled)
    alphas = np.power(d_flat / d_max, alpha_power)
    colors[:, -1] = alphas

    fig = plt.figure(figsize=(3.85, 3.15))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_flat, y_flat, z_flat, c=colors, s=point_size, edgecolor="none")

    # Mark max density
    max_idx = np.argmax(d_flat)
    mx, my, mz = x_flat[max_idx], y_flat[max_idx], z_flat[max_idx]

    latex_to_unicode = {r"$\phi$": "φ", r"$\xi$": "ξ", r"$\chi$": "χ"}
    converted = [latex_to_unicode[s] for s in labels]
    print(
        f"Max = [{converted[0]} {mx:.2f}, {converted[1]} {my:.2f}, {converted[2]} {mz:.2f}]"
    )

    # Draw projection lines
    for dim, (coord, lim) in enumerate(zip((mx, my, mz), grid_limits)):
        line = [[mx, mx], [my, my], [mz, mz]]
        line[dim][1] = lim[0]
        ax.plot(*line, color="black", linestyle="--", linewidth=1)

    configure_axes(ax, grid_limits, [f"{lbl} [°]" for lbl in labels])
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=1200)
    else:
        plt.show()


if __name__ == "__main__":
    import sys

    file = sys.argv[1]

    data = np.load(file)
    PDF = data["PDF"]
    coordinates = data["coordinates"]
    resolution = data["resolution"]
    grid = data["grid_limits"]
    labels = data["labels"]
    png_file = str(data["png_file"])

    plot_kde_3d(
        PDF,
        coordinates,
        grid,
        "torsion_distribution",
        labels,
        downsample=1,
        output_file=png_file,
    )
