import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ─── PCA ──────────────────────────────────────────────────────────────────────

def compute_pca(waveforms, all_chans=True, n_components=10):
    """
    waveforms : (n_spikes, n_channels, n_samples)
    all_chans : if True, flatten channels before PCA (recommended for cluster comparison)
    Returns pc_features (n_spikes, n_components) and fitted pca object(s)
    """
    n_spikes, n_chans, n_samples = waveforms.shape

    if all_chans:
        X = waveforms.reshape(n_spikes, -1)           # (n_spikes, n_chans * n_samples)
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X), pca
    else:
        pca_per_chan = []
        features_per_chan = np.empty((n_spikes, n_chans, n_components))
        for ch in range(n_chans):
            p = PCA(n_components=n_components)         # fresh PCA per channel
            features_per_chan[:, ch, :] = p.fit_transform(waveforms[:, ch, :])
            pca_per_chan.append(p)
        return features_per_chan, pca_per_chan


# ─── SPIKE TRAIN METRICS ──────────────────────────────────────────────────────

def firing_rate(spike_times_samples, duration_s, sample_rate=20000):
    return len(spike_times_samples) / duration_s


def presence_ratio(spike_times_samples, duration_s, sample_rate=20000, n_bins=100):
    st_s = spike_times_samples / sample_rate
    #print(st_s.min(), st_s.max())
    #print(duration_s)
    #print(st_s.max() / duration_s)
    counts, _ = np.histogram(st_s, bins=np.linspace(0, duration_s, n_bins + 1))
    return np.mean(counts > 0)                         # fraction of bins with ≥1 spike


def isi_violation_rate(spike_times_samples, sample_rate=20000, threshold_s=0.0015):
    """Returns violations / total ISIs — comparable across units."""
    st_s = np.sort(spike_times_samples) / sample_rate
    isi = np.diff(st_s)
    if len(isi) == 0:
        return np.nan
    return np.sum(isi < threshold_s) / len(isi)

# ─── CLUSTER QUALITY METRICS ──────────────────────────────────────────────────
def waveform_similarity(waveforms, threshold=0.85, use_median=False):
    """
    waveforms : (n_spikes, n_samples) array
    threshold : minimum correlation to template to count as similar
    Returns (fraction_above_threshold, per_spike_correlations)
    """
    template = np.median(waveforms, axis=0) if use_median else np.mean(waveforms, axis=0)

    # Vectorized Pearson correlation — shape only, amplitude-invariant
    w = waveforms - waveforms.mean(axis=1, keepdims=True)
    t = template - template.mean()

    correlations = (w @ t) / (np.linalg.norm(w, axis=1) * np.linalg.norm(t))

    fraction = np.mean(correlations >= threshold)
    return fraction, correlations


def _mahalanobis_to_cluster(unit_pcs, query_pcs):
    """
    Shared computation for isolation_distance and l_ratio.
    Returns squared Mahalanobis distances of query_pcs from unit_pcs distribution.
    """
    mu = unit_pcs.mean(axis=0)
    cov = np.cov(unit_pcs, rowvar=False)
    inv_cov = np.linalg.pinv(cov)                      # pinv handles near-singular cases
    diffs = query_pcs - mu
    return np.sum(diffs @ inv_cov * diffs, axis=1)     # (n_query,)


def isolation_distance(unit_pcs, other_pcs):
    """
    Mahalanobis distance at which the unit cluster contains as many
    background spikes as it has spikes of its own. Higher = better isolated.
    """
    n = len(unit_pcs)
    if len(other_pcs) < n:
        return np.nan
    mahal_sq = _mahalanobis_to_cluster(unit_pcs, other_pcs)
    return float(np.sort(mahal_sq)[n - 1])


def l_ratio(unit_pcs, other_pcs):
    """
    Cumulative Mahalanobis-based contamination, normalised by cluster size.
    Lower = less contamination. Companion metric to isolation_distance.
    """
    n_dims = unit_pcs.shape[1]
    mahal_sq = _mahalanobis_to_cluster(unit_pcs, other_pcs)
    # probability each background spike could belong to cluster
    L = np.sum(1.0 - chi2.cdf(mahal_sq, df=n_dims))
    return L / len(unit_pcs)


def d_prime(unit_pcs, other_pcs):
    """LDA-based d': optimal linear separation between unit and background."""
    X = np.vstack([unit_pcs, other_pcs])
    y = np.r_[np.ones(len(unit_pcs)), np.zeros(len(other_pcs))]
    lda = LinearDiscriminantAnalysis(n_components=1)
    z = lda.fit_transform(X, y).ravel()
    z_unit, z_other = z[y == 1], z[y == 0]
    return abs(z_unit.mean() - z_other.mean()) / np.sqrt(
        0.5 * (z_unit.var(ddof=1) + z_other.var(ddof=1))
    )


def nn_hit_rate(unit_pcs, other_pcs, k=5, seed=0):
    """
    Fraction of a unit's k-nearest neighbours that belong to the same unit.
    Subsamples other_pcs to match unit size to avoid density bias.
    """
    n = len(unit_pcs)
    rng = np.random.default_rng(seed)

    # balance: subsample other_pcs down to n if larger
    if len(other_pcs) > n:
        idx = rng.choice(len(other_pcs), size=n, replace=False)
        other_pcs = other_pcs[idx]

    X = np.vstack([unit_pcs, other_pcs])
    y = np.r_[np.ones(n), np.zeros(len(other_pcs))]

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    neighbor_idx = nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self

    unit_mask = y == 1
    hits = y[neighbor_idx[unit_mask]] == 1
    return float(hits.mean())


# ─── MAIN METRICS TABLE ───────────────────────────────────────────────────────

def compute_unit_metrics(
    unit_spike_times,       # dict: {unit_id: spike_times_in_samples}
    unit_pc_features,       # dict: {unit_id: (n_spikes, n_components) array}
    unit_channel_indicies,  # dict 
    unit_amps,              # dict 
    unit_waveforms,         # dict 
    duration_s,
    sorter,
    probe_geometry,
    sample_rate=20000,
    savefigs = False
):
    """
    Parameters
    ----------
    unit_spike_times  : {unit_id -> np.ndarray of spike times in samples}
    unit_pc_features  : {unit_id -> np.ndarray shape (n_spikes, n_pcs)}
    """
    # precompute full background pool once — don't rebuild inside the loop
    all_unit_ids = list(unit_pc_features.keys())
    all_pcs_stacked = {uid: unit_pc_features[uid] for uid in all_unit_ids}

    rows = []
    for unit_id in all_unit_ids:
        spike_times = np.asarray(unit_spike_times[unit_id])
        spike_amps = np.asarray(unit_amps[unit_id])
        unit_pcs = all_pcs_stacked[unit_id]
        unit_waves = np.asarray(unit_waveforms[unit_id])
        peak_chan_idx = unit_channel_indicies[unit_id]

        other_pcs = np.vstack([
            all_pcs_stacked[u] for u in all_unit_ids
            if u != unit_id and len(all_pcs_stacked[u]) > 0
        ])

        waveform_sim, corrs = waveform_similarity(unit_waves, 0.85, use_median=False)
        
        if savefigs:
            fig, ax = plt.subplots(2,1)
            fig.suptitle(f"{probe_geometry} unit {unit_id} (n={len(spike_times)})")
            ax[0].hist(corrs, bins=100)
            ax[0].set_title(f"waveform correlations (r >= {0.85}: {waveform_sim})")
            ax[1].hist(spike_amps, bins=100)
            ax[1].set_title(f"spike amps (mean: {np.mean(spike_amps)})")

            fig.tight_layout()
            plt.savefig(f"/Volumes/Trenholm2/neuroTechData/jun16_data/unit_figs/{probe_geometry}_{unit_id}")
            plt.close(fig)


        row = {
            "sorter":            sorter,
            "probe_geometry":    probe_geometry,
            "unit_id":           unit_id,
            "peak_chan_idx":     np.mode(peak_chan_idx),
            "n_spikes":          len(spike_times),
            "mean_spike_amp":    np.mean(spike_amps),
            "firing_rate_hz":    firing_rate(spike_times, duration_s, sample_rate),
            "presence_ratio":    presence_ratio(spike_times, duration_s, sample_rate),
            "waveform_similarity": waveform_sim,
            "isi_violation_rate": isi_violation_rate(spike_times, sample_rate),
            "isolation_distance": isolation_distance(unit_pcs, other_pcs),
            "l_ratio":           l_ratio(unit_pcs, other_pcs),
            "d_prime":           d_prime(unit_pcs, other_pcs),
            "nn_hit_rate":       nn_hit_rate(unit_pcs, other_pcs),
        }
        #row["log10_firing_rate"] = np.log10(row["firing_rate_hz"] + 1e-12)
        rows.append(row)

    return pd.DataFrame(rows)