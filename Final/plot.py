import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from pathlib import Path

# ── re-use from extract_data.py ──────────────────────────────────────────────
from extract_data import identical_spikes


def plot_bundled_unit_linear_assignment(
    bundled_res_mat_path,
    linear_res_mat_path,
    bundled_unit_id,
    tol=0,
    precomputed_matches=None,
    figsize=(10, 8),
    n_hist_bins=None,   # None = one bin per unique integer value (recommended)
    point_size=30,
    point_alpha=0.85,
    hist_alpha=0.35,
    hist_color="red",
    cmap="viridis",
):
    """
    For a given bundled JRCLUST unit, scatter-plot:

        X  =  linear cluster ID  (or −1 if that spike has no match in linear)
        Y  =  per-spike peak channel index  (spikeSites, 0-indexed)

    Points are coloured by local spike density (number of spikes sharing the
    same integer (linear_id, channel) coordinate).  A colour bar is shown on
    the right.  Marginal histograms in transparent red run along both axes.

    Parameters
    ----------
    bundled_res_mat_path  : str | Path
        Path to the bundled JRCLUST *_res.mat file.
    linear_res_mat_path   : str | Path
        Path to the linear JRCLUST *_res.mat file.
    bundled_unit_id       : int
        JRCLUST cluster id (1-based) of the bundled unit to inspect.
    tol                   : int
        Sample tolerance for spike matching (default 0 = exact same sample).
    precomputed_matches   : np.ndarray, shape (N, 2), optional
        Pre-computed output of identical_spikes(b_times, l_times, …).
        Column 0 = global bundled spike indices, column 1 = global linear
        spike indices.  If None the matches are recomputed from the files
        (slower for large recordings — pass them in if you already have them).
    figsize               : tuple
    n_hist_bins           : int | None
        Number of histogram bins.  None (default) uses one bin per unique
        integer level along each axis — usually the clearest choice.
    point_size            : int   marker size in points² (default 30)
    point_alpha           : float marker opacity (default 0.85)
    hist_alpha, hist_color  : marginal histogram aesthetics
    cmap                  : str   colormap for the density colour bar

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : (ax_main, ax_top, ax_right)
    data : dict with keys "linear_ids" and "peak_sites" (both per-spike arrays)
    """

    # ── 1. Load bundled arrays ────────────────────────────────────────────────
    with h5py.File(Path(bundled_res_mat_path), "r") as f:
        b_spike_clusters = f["spikeClusters"][...].squeeze().ravel().astype(np.int32)
        b_spike_times    = f["spikeTimes"][...].squeeze().ravel().astype(np.int64) - 1
        b_spike_sites    = f["spikeSites"][...].squeeze().ravel().astype(np.int32) - 1  # → 0-indexed

    # ── 2. Load linear arrays (only what we need) ─────────────────────────────
    with h5py.File(Path(linear_res_mat_path), "r") as f:
        l_spike_clusters = f["spikeClusters"][...].squeeze().ravel().astype(np.int32)
        l_spike_times    = f["spikeTimes"][...].squeeze().ravel().astype(np.int64) - 1

    # ── 3. Restrict to the requested bundled unit ─────────────────────────────
    unit_mask      = b_spike_clusters == bundled_unit_id   # (n_b_total,) bool
    unit_b_indices = np.where(unit_mask)[0]                # global indices into b arrays
    unit_b_sites   = b_spike_sites[unit_mask]              # per-spike peak channel index

    if len(unit_b_indices) == 0:
        raise ValueError(f"Bundled unit {bundled_unit_id} not found in {bundled_res_mat_path}")

    # ── 4. Compute (or reuse) spike matches ───────────────────────────────────
    if precomputed_matches is not None:
        matches = np.asarray(precomputed_matches, dtype=np.int64)
    else:
        print("Computing spike matches (pass precomputed_matches= to skip this)…")
        matches = identical_spikes(b_spike_times, l_spike_times, tol=tol)
        # shape (M, 2):  col-0 = global bundled idx,  col-1 = global linear idx

    # ── 5. Vectorised lookup: for each spike in the unit → linear cluster id ──
    #   We only care about matches whose bundled index falls inside this unit.
    linear_ids = np.full(len(unit_b_indices), -1, dtype=np.int32)

    if len(matches) > 0:
        # Sort matches by bundled index for searchsorted
        order          = np.argsort(matches[:, 0])
        sorted_b_idx   = matches[order, 0]
        sorted_l_idx   = matches[order, 1]

        # For each unit spike, find its position in sorted_b_idx
        ins = np.searchsorted(sorted_b_idx, unit_b_indices)

        # Clamp to valid range and verify the hit is exact
        ins_clamped = np.clip(ins, 0, len(sorted_b_idx) - 1)
        hit_mask    = sorted_b_idx[ins_clamped] == unit_b_indices  # exact match only

        matched_l_indices = sorted_l_idx[ins_clamped[hit_mask]]
        linear_ids[hit_mask] = l_spike_clusters[matched_l_indices]

    # ── 6. Per-point density  ─────────────────────────────────────────────────
    #   Both axes are discrete integers, so density = spike count per (x, y) cell.
    x = linear_ids    # shape (n_unit_spikes,)   values: -1 or linear cluster id
    y = unit_b_sites  # shape (n_unit_spikes,)   values: 0-indexed channel index

    unique_x = np.sort(np.unique(x))
    unique_y = np.sort(np.unique(y))

    x_bins_d = np.arange(unique_x[0] - 0.5, unique_x[-1] + 1.5)
    y_bins_d = np.arange(unique_y[0] - 0.5, unique_y[-1] + 1.5)

    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins_d, y_bins_d])

    # Map each spike to its 2-D bin count
    xi = np.clip(np.searchsorted(xedges[1:], x, side="left"), 0, H.shape[0] - 1)
    yi = np.clip(np.searchsorted(yedges[1:], y, side="left"), 0, H.shape[1] - 1)
    density = H[xi, yi]

    # ── 7. Build figure ───────────────────────────────────────────────────────
    #   GridSpec: 2 rows × 3 cols
    #     [top-hist]  [empty]    [empty]
    #     [scatter ]  [r-hist]   [colorbar]
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[5, 1, 0.25],
        height_ratios=[1, 5],
        hspace=0.04,
        wspace=0.08,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar  = fig.add_subplot(gs[1, 2])

    # ── scatter coloured by density ───────────────────────────────────────────
    # Sort by density ascending so densest points render on top
    order = np.argsort(density)
    sc = ax_main.scatter(
        x[order], y[order],
        c=density[order],
        cmap=cmap,
        s=point_size,
        alpha=point_alpha,
        linewidths=0,
        rasterized=True,
    )
    ax_main.set_xlabel("Linear cluster ID  (−1 = unmatched bundled spike)")
    ax_main.set_ylabel("Per-spike peak channel index  (0-indexed)")

    from matplotlib.ticker import MultipleLocator

    ax_main.set_axisbelow(True)

    # major grid: existing tick locations
    ax_main.grid(True, which="major", color="lightgrey",
                linewidth=0.6, linestyle="--", alpha=0.8)

    # minor x ticks every 5
    ax_main.xaxis.set_minor_locator(MultipleLocator(2))

    # finer vertical gridlines every 5
    ax_main.grid(True, which="minor", axis="x", color="lightgrey",
                linewidth=0.35, linestyle=":", alpha=0.7)


    #ax_main.grid(True, which="both", color="lightgrey", linewidth=0.6, linestyle="--", alpha=0.7)

    # Mark the -1 column visually
    if np.any(x == -1):
        ax_main.axvline(-1, color="salmon", linewidth=0.8, linestyle="--", alpha=0.6)

    # ── colour bar ────────────────────────────────────────────────────────────
    cbar = fig.colorbar(sc, cax=ax_cbar)
    cbar.set_label("Spikes per cell", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── top histogram  (distribution over linear cluster IDs) ─────────────────
    if n_hist_bins is None:
        x_bins = np.arange(unique_x[0] - 0.5, unique_x[-1] + 1.5)
    else:
        x_bins = n_hist_bins

    ax_top.hist(x, bins=x_bins, color=hist_color, alpha=hist_alpha, edgecolor="none")
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.set_ylabel("Count", fontsize=8)
    ax_top.spines[["top", "right"]].set_visible(False)

    # ── right histogram  (distribution over peak channel index) ───────────────
    if n_hist_bins is None:
        y_bins = np.arange(unique_y[0] - 0.5, unique_y[-1] + 1.5)
    else:
        y_bins = n_hist_bins

    ax_right.hist(y, bins=y_bins, orientation="horizontal",
                  color=hist_color, alpha=hist_alpha, edgecolor="none")
    ax_right.tick_params(labelleft=False, left=False)
    ax_right.set_xlabel("Count", fontsize=8)
    ax_right.spines[["top", "right"]].set_visible(False)

    # ── title ─────────────────────────────────────────────────────────────────
    n_matched   = int(np.sum(x > 0))
    n_unmatched = int(np.sum(x == -1))
    n_linear    = len(np.unique(x[x > 0]))
    fig.suptitle(
        f"Bundled unit B{bundled_unit_id}  │  {len(x):,} spikes  │  "
        f"{n_matched:,} matched to {n_linear} linear cluster(s)  │  "
        f"{n_unmatched:,} unmatched (−1)",
        fontsize=10,
    )

    return fig, (ax_main, ax_top, ax_right), {"linear_ids": linear_ids, "peak_sites": unit_b_sites}




# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import h5py
    from pathlib import Path

    linear_res  = r"/Volumes/Trenholm2/neuroTechData/jun16_data/Viktor_08_sortings/linear_5_5_JRCLUST/linear_5_5_amp_res.mat"
    bundled_res = r"/Volumes/Trenholm2/neuroTechData/jun16_data/Viktor_08_sortings/bundled_5_5_JRCLUST/bundled_5_5_amp_res.mat"

    # Optional: pre-compute matches once and reuse across multiple unit calls
    # (avoids re-running identical_spikes for every unit)
    #
    with h5py.File(bundled_res, "r") as f:
        b_times = f["spikeTimes"][...].squeeze() - 1
        b_spike_sites    = f["spikeSites"][...].squeeze().ravel().astype("int32") - 1
    with h5py.File(linear_res, "r") as f:
        l_times = f["spikeTimes"][...].squeeze() - 1
        l_spike_sites    = f["spikeSites"][...].squeeze().ravel().astype("int32") - 1

    matches = identical_spikes(b_times, l_times, tol=0, channels_A=b_spike_sites, channels_B=l_spike_sites)

    for i in range(45):
        fig, axes, data = plot_bundled_unit_linear_assignment(
            bundled_res_mat_path=bundled_res,
            linear_res_mat_path=linear_res,
            bundled_unit_id=i+1,
            tol=0,
            precomputed_matches=matches,   # pass here once computed
        )
        plt.show()