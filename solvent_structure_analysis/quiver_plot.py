from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# PLOT CONTROL

PLOT_CONFIG = {
    # downsampling
    "step": 2,
    # arrow appearance
    "length": 1.0,
    "alpha": 0.2,
    "linewidth": 1,
    "arrow_ratio": 0.2,
    # coloring
    "cmap": "coolwarm",
    "color_by": "z",  # options: "z", "distance"
    # axis limits (None = auto)
    "xlim": [-4, 4],
    "ylim": [-4, 4],
    "zlim": [-4, 4],
    # slicing (None = disabled)
    # "slice": None,
    "slice": {
        "axis": "x",  # "x", "y", "z"
        "min": -0.8,
        "max": 0.8,
    },
}

# -------


def apply_spatial_filter(pos, vec, config):
    mask = np.ones(len(pos), dtype=bool)

    # --- slicing ---
    sl = config["slice"]
    if sl is not None:
        axis_map = {"x": 0, "y": 1, "z": 2}
        ax_idx = axis_map[sl["axis"]]

        mask &= (pos[:, ax_idx] >= sl["min"]) & (pos[:, ax_idx] <= sl["max"])

    # --- manual limits (hard filtering, not just view) ---
    for i, key in enumerate(["xlim", "ylim", "zlim"]):
        lim = config[key]
        if lim is not None:
            mask &= (pos[:, i] >= lim[0]) & (pos[:, i] <= lim[1])

    return pos[mask], vec[mask]


def downsample(pos, vec, step):
    idx = np.random.choice(len(pos), size=int(len(pos) / step), replace=False)
    return pos[idx], vec[idx]


if __name__ == "__main__":
    INPUT_FILE = "solvent_vectors_data.npz"

    data = np.load(INPUT_FILE)

    # reconstruct structure
    all_results = {}

    for key in data.files:
        print(key)
        if key.startswith("pos_"):
            resid = int(key.split("_")[1])

            pos = data[f"pos_{resid}"]
            vec = data[f"vec_{resid}"]

            all_results[resid] = {
                "positions": pos,
                "vectors": vec,
            }

    print(f"[LOAD] Loaded residues: {list(all_results.keys())}")

    for resid, data in all_results.items():
        positions = np.array(data["positions"])
        vectors = np.array(data["vectors"])

        # --- filtering ---
        pos, vec = apply_spatial_filter(positions, vectors, PLOT_CONFIG)

        # --- downsampling ---
        pos, vec = downsample(pos, vec, PLOT_CONFIG["step"])

        if len(pos) == 0:
            print(f"[WARN] No data after filtering for resid {resid}")
            continue

        # --- coloring ---
        vec_norm = vec / np.linalg.norm(vec, axis=1, keepdims=True)

        if PLOT_CONFIG["color_by"] == "z":
            scalars = vec_norm[:, 2]
            norm = mcolors.Normalize(vmin=-1, vmax=1)
        elif PLOT_CONFIG["color_by"] == "distance":
            scalars = np.linalg.norm(pos, axis=1)
            norm = mcolors.Normalize(vmin=0, vmax=max(positions))
        else:
            print("Wrong color scheme: ", PLOT_CONFIG["color_by"])
            exit()

        cmap = cm.get_cmap(PLOT_CONFIG["cmap"])

        # --- plot ---
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        for p, v, s in zip(pos, vec, scalars):
            color = cmap(norm(s))
            ax.quiver(
                p[0],
                p[1],
                p[2],
                v[0],
                v[1],
                v[2],
                length=PLOT_CONFIG["length"],
                normalize=True,
                color=color,
                arrow_length_ratio=PLOT_CONFIG["arrow_ratio"],
                alpha=PLOT_CONFIG["alpha"],
                linewidth=PLOT_CONFIG["linewidth"],
            )

        # urethane axis
        ax.quiver(0, 0, 0, 0, 0, 2, color="black", linewidth=2)

        # --- axis limits (view only) ---
        if PLOT_CONFIG["xlim"]:
            ax.set_xlim(PLOT_CONFIG["xlim"])
        if PLOT_CONFIG["ylim"]:
            ax.set_ylim(PLOT_CONFIG["ylim"])
        if PLOT_CONFIG["zlim"]:
            ax.set_ylim(PLOT_CONFIG["zlim"])

        # labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # title includes slice info
        sl = PLOT_CONFIG["slice"]
        if sl:
            title_slice = f"{sl['axis']} ∈ [{sl['min']}, {sl['max']}]"
        else:
            title_slice = "full volume"

        ax.set_title(f"Resid {resid} | {title_slice}")

        # colorbar
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        plt.colorbar(mappable, ax=ax, pad=0.1)

        plt.savefig(f"3d_vectors_resid_{resid}.png", dpi=300)
        plt.close()
