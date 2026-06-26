import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors

def compute_PCA():

    return 0 


def firing_rate(spike_times, duration_s):
    return len(spike_times) / duration_s


def presence_ratio(spike_times, duration_s, bin_size_s=60):
    bins = np.arange(0, duration_s + bin_size_s, bin_size_s)
    counts, _ = np.histogram(spike_times, bins=bins)
    return np.mean(counts > 0)


def isi_violations(spike_times, threshold_s=0.0015):
    isi = np.diff(np.sort(spike_times))
    return np.sum(isi < threshold_s)

def isolation_distance(unit_pcs, other_pcs):
    n = len(unit_pcs)
    if len(other_pcs) < n:
        return np.nan

    mu = unit_pcs.mean(axis=0)
    cov = np.cov(unit_pcs, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    diffs = other_pcs - mu
    mahal = np.sum(diffs @ inv_cov * diffs, axis=1)
    return np.sort(mahal)[n - 1]


def d_prime(unit_pcs, other_pcs):
    X = np.vstack([unit_pcs, other_pcs])
    y = np.r_[np.ones(len(unit_pcs)), np.zeros(len(other_pcs))]

    lda = LinearDiscriminantAnalysis(n_components=1)
    z = lda.fit_transform(X, y).ravel()

    z_unit = z[y == 1]
    z_other = z[y == 0]

    return abs(z_unit.mean() - z_other.mean()) / np.sqrt(
        0.5 * (z_unit.var(ddof=1) + z_other.var(ddof=1))
    )


def nn_hit_rate(unit_pcs, other_pcs, k=5):
    X = np.vstack([unit_pcs, other_pcs])
    y = np.r_[np.ones(len(unit_pcs)), np.zeros(len(other_pcs))]

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neighbors = nn.kneighbors(X, return_distance=False)[:, 1:]

    unit_idx = np.where(y == 1)[0]
    hits = y[neighbors[unit_idx]] == 1

    return hits.mean()

def compute_unit_metrics(
    units,
    duration_s,
    sorter,
    probe_geometry,
    pc_features=None,
    amplitudes=None,
    mean_waveforms=None,
):
    rows = []

    for unit_id, spike_times in units.items():
        spike_times = np.asarray(spike_times)
        row = {
            "sorter": sorter,
            "probe_geometry": probe_geometry,
            "unit_id": unit_id,
            "n_spikes": len(spike_times),
            "firing_rate_hz": firing_rate(spike_times, duration_s),
            "presence_ratio": presence_ratio(spike_times, duration_s),
            "isi_violations_count": isi_violations(spike_times),
        }

        row["log10_firing_rate"] = np.log10(row["firing_rate_hz"] + 1e-12)


        if pc_features is not None:
            unit_pcs = pc_features[unit_id]
            other_pcs = np.vstack([
                pc_features[u] for u in pc_features
                if u != unit_id and len(pc_features[u]) > 0
            ])

            row["isolation_distance"] = isolation_distance(unit_pcs, other_pcs)
            row['L-ratio'] = None
            row["d_prime"] = d_prime(unit_pcs, other_pcs)
            row["nn_hit_rate"] = nn_hit_rate(unit_pcs, other_pcs)

        rows.append(row)

    return pd.DataFrame(rows)