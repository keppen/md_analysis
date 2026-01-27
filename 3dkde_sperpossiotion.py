# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch


def configure_axes(ax, limits, labels, num_ticks=3, fontsize=8, pad=2):
    """
    Apply consistent styling to a 3D Axes object.
    """
    # Set limits and labels
    for dim, lim in enumerate(limits):
        axis_char = "xyz"[dim]
        getattr(ax, f"set_{axis_char}lim")(lim)
        getattr(ax, f"set_{axis_char}label")(
            labels[dim], fontsize=fontsize, labelpad=pad, rotation=[180, 180, 180][dim]
        )
        ax.tick_params(axis=axis_char, labelsize=8)
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


def _prepare_flat(PDF, coords, downsample=1, threshold=1e-12):
    """Flatten and mask tiny values; returns x,y,z,d and grid slices for projections."""
    pdf_ds = PDF[::downsample, ::downsample, ::downsample].astype(np.float32)
    coords_ds = coords[:, ::downsample, ::downsample, ::downsample].astype(np.float32)
    pdf_ds = pdf_ds + threshold
    x, y, z = coords_ds
    x_flat, y_flat, z_flat = x.ravel(), y.ravel(), z.ravel()
    d_flat = pdf_ds.ravel()
    mask = d_flat > threshold
    return x_flat[mask], y_flat[mask], z_flat[mask], d_flat[mask], coords_ds


def plot_isosurface_intersection(
    file1,
    file2,
    threshold=3e-7,
    color1=(0.0, 0.25, 0.8, 0.2),  # vivid blue
    color2=(0.0, 0.35, 0.0, 0.3),  # vivid green
    color_inter=(0.8, 0.1, 0.6, 0.3),
    show=False,
    output_file=None,
    figsize=(3.85, 3.15),
):
    """
    Plot isosurfaces for two volumetric datasets and their intersection.
    Isosurface for each dataset is at `level_ratio * max_density` (default 0.5).
    Intersection surface is produced from the binary intersection mask.

    Parameters
    ----------
    file1, file2 : str or path-like
        Paths to .npz files containing keys: 'PDF', 'coordinates', 'grid_limits', 'labels'.
    level_ratio : float
        Fraction of each dataset's max density used as the isosurface level (0..1).
    color1, color2, color_inter : RGBA tuples
        Colors for dataset A, dataset B and their intersection respectively.
    show : bool
        Whether to call plt.show() at the end.
    output_file : str or None
        If provided, save the figure to this path.
    figsize : tuple
        Figure size passed to plt.figure().

    Returns
    -------
    metrics : dict
        Dictionary with numeric overlap metrics: voxel counts, volumes, Dice coefficient.
    """
    try:
        from skimage import measure
    except Exception as exc:
        raise ImportError(
            "This function requires scikit-image. Install with `pip install scikit-image`."
        ) from exc

    # --- load data ---------------------------------------------------------
    data1 = np.load(file1)
    data2 = np.load(file2)
    labels1 = ["φ", "ξ", "χ"]
    grid1 = [[-180, 180], [-180, 180], [-180, 180]]
    labels2 = ["φ", "ξ", "χ"]
    grid1 = [[-180, 180], [-180, 180], [-180, 180]]

    # Extract
    PDF1, coords1 = (
        data1["data"],
        data1["coords"],
    )
    PDF2, coords2 = (
        data2["data"],
        data2["coords"],
    )

    if PDF1.shape != PDF2.shape:
        # It's still possible to proceed if shapes differ, but simplest route is to require equal shapes here.
        raise ValueError(
            "PDF shapes must match for this function (got %s and %s)."
            % (PDF1.shape, PDF2.shape)
        )

    # --- compute thresholds ------------------------------------------------
    # thr1 = float(level_ratio * np.nanmax(PDF1))
    # thr2 = float(level_ratio * np.nanmax(PDF2))
    # print(thr2 * level_ratio)
    thr1, thr2 = threshold, threshold

    # --- build boolean masks at thresholds ---------------------------------
    mask1 = PDF1 >= thr1
    mask2 = PDF2 >= thr2
    mask_inter = mask1 & mask2
    mask_a_only = mask1 & (~mask2)
    mask_b_only = mask2 & (~mask1)

    # --- get spacing and origin from coordinates (assumes regular grid) ----
    # coords shape assumed (3, nx, ny, nz)
    x_coords = coords1[0][:, 0, 0].astype(float)
    y_coords = coords1[1][0, :, 0].astype(float)
    z_coords = coords1[2][0, 0, :].astype(float)

    # spacing: mean delta in each axis (works for uniform grids)
    dx = float(np.mean(np.diff(x_coords)))
    dy = float(np.mean(np.diff(y_coords)))
    dz = float(np.mean(np.diff(z_coords)))
    spacing = (dx, dy, dz)
    origin = (float(x_coords[0]), float(y_coords[0]), float(z_coords[0]))

    # marching_cubes expects volumes in (dim0, dim1, dim2) voxel order: that's our PDF shape
    # Use original PDFs for dataset surfaces, and binary masks for intersection.
    # marching_cubes returns vertices in voxel coordinates; scaling with spacing and adding origin yields world coords.
    def extract_mesh(volume, level, spacing, origin):
        """
        Returns verts_world (N,3) and faces (M,3) extracted at `level`.
        """
        try:
            verts, faces, normals, values = measure.marching_cubes(
                volume.astype(np.float32), level=level, spacing=spacing
            )
        except AttributeError:
            # older scikit-image naming
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume.astype(np.float32), level=level, spacing=spacing
            )
        # marching_cubes with spacing already scales vertices, but the returned coords start at 0 index.
        # So add the origin offset to get actual world coordinates:
        verts_world = verts + np.array(origin, dtype=float)
        return verts_world, faces

    # --- extract meshes ----------------------------------------------------
    # For the datasets we use the continuous PDF at threshold thr1/thr2
    # For the intersection, use the binary mask as a float volume with level 0.5
    verts1, faces1 = extract_mesh(PDF1, level=thr1, spacing=spacing, origin=origin)
    verts2, faces2 = extract_mesh(PDF2, level=thr2, spacing=spacing, origin=origin)

    # If intersection is empty, marching_cubes may fail; guard for empty case
    if np.any(mask_inter):
        verts_inter, faces_inter = extract_mesh(
            mask_inter.astype(np.uint8), level=0.5, spacing=spacing, origin=origin
        )
    else:
        verts_inter, faces_inter = None, None

    # --- compute voxel-based volumes & dice --------------------------------
    voxel_volume = abs(dx * dy * dz)
    vox1 = int(mask1.sum())
    vox2 = int(mask2.sum())
    vox_inter = int(mask_inter.sum())

    vol1 = vox1 * voxel_volume
    vol2 = vox2 * voxel_volume
    vol_inter = vox_inter * voxel_volume

    dice = (2.0 * vox_inter) / (vox1 + vox2) if (vox1 + vox2) > 0 else 0.0

    # --- plotting ----------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((np.ptp(x_coords), np.ptp(y_coords), np.ptp(z_coords)))

    def add_poly_collection(verts, faces, facecolor, edgecolor=None, linewidth=0.0):
        """Create Poly3DCollection from marching_cubes verts+faces and add to ax."""
        mesh_tris = verts[faces]  # (nfaces, 3, 3)
        coll = Poly3DCollection(
            mesh_tris, facecolors=facecolor, linewidths=linewidth, edgecolors=edgecolor
        )
        coll.set_alpha(facecolor[3] if len(facecolor) == 4 else None)
        ax.add_collection3d(coll)
        return coll

    # Dataset A surface
    add_poly_collection(verts1, faces1, color1, edgecolor=None, linewidth=0.0)

    # Dataset B surface
    add_poly_collection(verts2, faces2, color2, edgecolor=None, linewidth=0.0)

    # --- intersection shadows using scatter on 3 faces --------------------
    if np.any(mask_inter):
        inter = mask_inter.astype(float)

        # Projections: 1 where intersection touches that plane
        proj_xy = np.max(inter, axis=2)  # XY plane (top)
        proj_xz = np.max(inter, axis=1)  # XZ plane (front)
        proj_yz = np.max(inter, axis=0)  # YZ plane (side)

        # Coordinates for projections
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        Xz, Zx = np.meshgrid(x_coords, z_coords, indexing="ij")
        Yz, Zy = np.meshgrid(y_coords, z_coords, indexing="ij")

        shadow_color = (0.1, 0.1, 0.1, 0.02)  # dark grey, semi-transparent

        # Top face (XY, max Z)
        mask_xy = proj_xy > 0
        ax.scatter(
            X[mask_xy],
            Y[mask_xy],
            np.full(mask_xy.sum(), z_coords[0]),
            color=shadow_color,
            s=1,
            depthshade=False,
        )

        # Front face (XZ, min Y)
        mask_xz = proj_xz > 0
        ax.scatter(
            Xz[mask_xz],
            np.full(mask_xz.sum(), y_coords[0]),
            Zx[mask_xz],
            color=shadow_color,
            s=1,
            depthshade=False,
        )

        # Side face (YZ, min X)
        mask_yz = proj_yz > 0
        ax.scatter(
            np.full(mask_yz.sum(), x_coords[0]),
            Yz[mask_yz],
            Zy[mask_yz],
            color=shadow_color,
            s=1,
            depthshade=False,
        )

    # Intersection (if present) - draw on top with strong color/alpha
    # if verts_inter is not None and faces_inter is not None and len(faces_inter) > 0:
    #     add_poly_collection(
    #         verts_inter, faces_inter, color_inter, edgecolor=None, linewidth=0.0
    #     )
    # else:
    #     # no intersection surface found (empty intersection), optionally note this
    #     pass

    # create legend proxies
    proxies = [
        Patch(
            facecolor=color1,
            edgecolor="none",
            label=f"BuOH, cluster 1",
        ),
        Patch(
            facecolor=color2,
            edgecolor="none",
            label=r"$2.6_{14}$ helix",
        ),
    ]
    if verts_inter is not None and faces_inter is not None and len(faces_inter) > 0:
        proxies.append(
            Patch(
                facecolor=[0.1, 0.1, 0.1, 0.3],
                edgecolor="none",
                label=r"Intersection projections",
            ),
        )

    # ax.legend(
    #     handles=proxies,
    #     fontsize=8,
    #     frameon=False,
    #     loc="upper right",
    #     bbox_to_anchor=(1.5, 1.5),
    # )

    configure_axes(ax, grid1, [f"{lbl} [°]" for lbl in labels1])
    # set limits from grid limits (if available) to tightly bound the plot
    # try:
    #     xmin, xmax = grid1[0]
    #     ymin, ymax = grid1[1]
    #     zmin, zmax = grid1[2]
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
    #     ax.set_zlim(zmin, zmax)
    # except Exception:
    #     pass

    plt.tight_layout()
    if output_file:
        plt.savefig(
            output_file,
            dpi=1200,
            # bbox_inches="tight",
        )
    if show:
        plt.show()
    plt.close(fig)

    metrics = {
        "voxels_A": vox1,
        "voxels_B": vox2,
        "voxels_intersection": vox_inter,
        "volume_A": vol1,
        "volume_B": vol2,
        "volume_intersection": vol_inter,
        "dice_coefficient": float(dice),
        "threshold_A": float(thr1),
        "threshold_B": float(thr2),
    }
    # also print concise summary
    print("Isosurface intersection summary:")
    print(f"  voxels A: {vox1}, voxels B: {vox2}, vox inter: {vox_inter}")
    print(
        f"  volumes (same units as coords): A={vol1:.4g}, B={vol2:.4g}, inter={vol_inter:.4g}"
    )
    print(f"  Dice (binary masks at thr): {dice:.4f}")
    return metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python script.py <file1.npz> <file2.npz> [output.png]")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None

    plot_isosurface_intersection(file1, file2, output_file=output)
