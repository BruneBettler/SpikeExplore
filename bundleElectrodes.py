import numpy as np
import matplotlib.pyplot as plt


def bundle_electrodes(electrode, bundle_width=40.0, mode="tanh", alpha=3.0, plot=True):
    """
    Collapse electrode x-positions into a bundle of specified width.

    Parameters
    ----------
    electrode : array-like, shape (N, 2)
        Column 0 = x positions (um), column 1 = y positions (um).
    bundle_width : float, default=40.0
        Total x-width (um) of the final bundle.
    mode : {"linear", "tanh"}, default="tanh"
        - "linear": uniform x scaling into the bundle width
        - "tanh": non-uniform compression; center moves less, edges compress more
    alpha : float, default=3.0
        Non-uniformity parameter for "tanh" mode.
        Larger alpha preserves central spacing more strongly.
    plot : bool, default=True
        Whether to generate diagnostic plots.

    Returns
    -------
    electrode_bundle : ndarray, shape (N, 2)
        Bundled electrode positions.
    dx : ndarray, shape (N,)
        Lateral displacement for each electrode (x_bundle - x0).
    info : dict
        Useful diagnostic quantities.
    """
    electrode = np.asarray(electrode, dtype=float)

    if electrode.ndim != 2 or electrode.shape[1] < 2:
        raise ValueError("`electrode` must be an array of shape (N, 2) or greater.")

    x0 = electrode[:, 0]
    y0 = electrode[:, 1]

    x_min, x_max = x0.min(), x0.max()
    x_center = 0.5 * (x_min + x_max)
    x_extent = x_max - x_min
    half_ext = x_extent / 2.0
    half_bw = bundle_width / 2.0

    if half_ext == 0:
        raise ValueError("All x positions are identical, so bundling is undefined.")

    u = (x0 - x_center) / half_ext  # normalized x in [-1, 1]

    mode = mode.lower()
    if mode == "linear":
        x_bundle = x_center + u * half_bw
    elif mode == "tanh":
        x_bundle = x_center + half_bw * np.tanh(alpha * u) / np.tanh(alpha)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'linear' or 'tanh'.")

    y_bundle = y0.copy()
    dx = x_bundle - x0
    electrode_bundle = np.column_stack((x_bundle, y_bundle))

    # Diagnostics
    order = np.argsort(x0)
    x0_s = x0[order]
    xb_s = x_bundle[order]
    d0 = np.diff(x0_s)
    db = np.diff(xb_s)

    info = {
        "x_center": x_center,
        "original_x_extent": x_extent,
        "bundled_x_extent": x_bundle.max() - x_bundle.min(),
        "max_abs_dx": np.abs(dx).max(),
        "min_abs_dx": np.abs(dx).min(),
        "design_spacing_min": d0.min() if d0.size else np.nan,
        "design_spacing_median": np.median(d0) if d0.size else np.nan,
        "design_spacing_max": d0.max() if d0.size else np.nan,
        "bundled_spacing_min": db.min() if db.size else np.nan,
        "bundled_spacing_median": np.median(db) if db.size else np.nan,
        "bundled_spacing_max": db.max() if db.size else np.nan,
    }

    print(f"\nOriginal x extent : {info['original_x_extent']:.2f} um  (center {x_center:.2f})")
    print(f"Bundle   x extent : {info['bundled_x_extent']:.2f} um")
    print(f"Max |dx|          : {info['max_abs_dx']:.2f} um  (lateral fiber pulled hardest)")
    print(f"Min |dx|          : {info['min_abs_dx']:.2f} um  (most-central fiber moved least)")

    print("\nAdjacent-contact spacing (sorted by x):")
    print(
        f"  design   : min {info['design_spacing_min']:.2f}  "
        f"median {info['design_spacing_median']:.2f}  "
        f"max {info['design_spacing_max']:.2f} um"
    )
    print(
        f"  bundled  : min {info['bundled_spacing_min']:.2f}  "
        f"median {info['bundled_spacing_median']:.2f}  "
        f"max {info['bundled_spacing_max']:.2f} um"
    )

    if plot:
        _plot_bundle_diagnostics(
            x0, y0, x_bundle, y_bundle, x_center, bundle_width, dx, mode
        )

    return electrode_bundle, dx, info


def _plot_bundle_diagnostics(x0, y0, x_bundle, y_bundle, x_center, bundle_width, dx, mode):
    """Internal helper for plotting."""
    half_bw = bundle_width / 2.0
    y_pad = 20.0
    y_span = y0.max() - y0.min()
    dy_plot = y_span + 100.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].plot(
        x0, y0, "o",
        markerfacecolor=(0.2, 0.4, 0.9),
        markeredgecolor="k",
        markersize=6
    )
    axes[0].axvline(x_center, linestyle="--", color="k")
    axes[0].set_title(f"Original (span = {x0.max() - x0.min():.1f} um)")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[0].axis("equal")
    axes[0].grid(True)

    # Bundled
    axes[1].plot(
        x_bundle, y_bundle, "s",
        markerfacecolor=(0.9, 0.3, 0.3),
        markeredgecolor="k",
        markersize=6
    )
    rect = plt.Rectangle(
        (x_center - half_bw, y0.min() - y_pad),
        bundle_width,
        y_span + 2 * y_pad,
        fill=False,
        linestyle="--",
        edgecolor="k"
    )
    axes[1].add_patch(rect)
    axes[1].set_title(f"Bundled ({bundle_width:g} um, mode={mode})")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    axes[1].axis("equal")
    axes[1].grid(True)

    # Fiber trajectories
    for xi, yi, xb in zip(x0, y0, x_bundle):
        axes[2].plot([xi, xb], [yi, yi + dy_plot], "-", color=(0.5, 0.5, 0.5, 0.6))
    axes[2].plot(
        x0, y0, "o",
        markerfacecolor=(0.2, 0.4, 0.9),
        markeredgecolor="k",
        markersize=5
    )
    axes[2].plot(
        x_bundle, y0 + dy_plot, "s",
        markerfacecolor=(0.9, 0.3, 0.3),
        markeredgecolor="k",
        markersize=5
    )
    axes[2].set_title("Fiber trajectories (schematic)")
    axes[2].set_xlabel("x (um)")
    axes[2].set_ylabel("y (um)")
    axes[2].axis("equal")
    axes[2].grid(True)

    plt.tight_layout()

    # Displacement figure
    plt.figure(figsize=(7, 4))
    plt.plot(
        x0, np.abs(dx), "o-",
        markerfacecolor=(0.2, 0.7, 0.3),
        markeredgecolor="k",
        linewidth=1.2
    )
    plt.xlabel("Original x (um)")
    plt.ylabel("|dx| (um)")
    plt.title("How far each fiber is pulled inward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    # Electrode: shape (N, 2)
    # column 0 = x, column 1 = y
    Electrode = np.array([
        [0.0,   0.0],
        [20.0, 40.0],
        [40.0, 80.0],
        [60.0, 120.0],
    ])

    Electrode_bundle, dx, info = bundle_electrodes(
        Electrode,
        bundle_width=40,
        mode="tanh",
        alpha=3.0,
        plot=True,
    )