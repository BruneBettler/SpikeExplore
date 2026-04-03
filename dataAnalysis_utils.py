import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.fft import rfft, irfft


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def _autocorr_fft(x: np.ndarray, max_lag_samples: int) -> np.ndarray:
    """
    Fast autocorrelation using FFT.

    Parameters
    ----------
    x : array, shape (n_samples, n_channels)
        Mean-centered signals.
    max_lag_samples : int
        Maximum lag to return.

    Returns
    -------
    acf : array, shape (max_lag_samples + 1, n_channels)
        Normalized autocorrelation, acf[0] = 1.
    """
    n_samples, n_channels = x.shape
    nfft = _next_pow2(2 * n_samples - 1)

    fx = rfft(x, n=nfft, axis=0)
    acf = irfft(fx * np.conj(fx), n=nfft, axis=0)[:max_lag_samples + 1]

    # Unbiased normalization by overlap count
    overlap = np.arange(n_samples, n_samples - max_lag_samples - 1, -1, dtype=np.float64)[:, None]
    acf = acf / overlap

    # Normalize so acf[0] = 1
    acf0 = acf[0:1]
    good = acf0 > 0
    acf[:, good[0]] /= acf0[:, good[0]]
    return acf


def find_repetitive_channels(
    recording,
    segment_index: int = 0,
    start_time_s: float = 0.0,
    duration_s: float = 20.0,
    freq_band: tuple[float, float] = (1.0, 300.0),
    nperseg: int | None = None,
    peak_ratio_threshold: float = 8.0,
    peak_prominence_ratio: float = 4.0,
    require_acf: bool = True,
    acf_min_period_s: float = 1 / 80.0,
    acf_max_period_s: float = 1 / 1.0,
    acf_threshold: float = 0.10,
    max_plots: int = 24,
    plot_raw_seconds: float = 1.0,
):
    """
    Quickly detect channels with clear repetitive structure in a SpikeInterface recording.

    Strategy
    --------
    1) Compute PSD for all channels at once using Welch.
    2) Flag channels with a strong narrow-band spectral peak.
    3) Optionally confirm using FFT-based autocorrelation.
    4) Plot only flagged channels.

    Parameters
    ----------
    recording : spikeinterface RecordingExtractor
    segment_index : int
    start_time_s : float
    duration_s : float
        Analyze only this chunk for speed.
    freq_band : (float, float)
        Frequency band in which to look for repetitive structure.
    nperseg : int or None
        Welch window length. If None, chosen automatically.
    peak_ratio_threshold : float
        Require max PSD / median PSD within freq_band >= this.
    peak_prominence_ratio : float
        Require spectral peak prominence / median PSD >= this.
    require_acf : bool
        If True, also require an autocorrelation peak.
    acf_min_period_s : float
        Shortest period to consider in ACF confirmation.
    acf_max_period_s : float
        Longest period to consider in ACF confirmation.
    acf_threshold : float
        Minimum ACF peak height after lag 0.
    max_plots : int
        Maximum number of flagged channels to plot.
    plot_raw_seconds : float
        Raw trace duration to show for each plotted channel.

    Returns
    -------
    result : dict
        Contains flagged channel ids, scores, PSDs, frequencies, etc.
    """
    fs = recording.get_sampling_frequency()
    channel_ids = np.array(recording.get_channel_ids())
    n_channels = recording.get_num_channels()

    start_frame = int(round(start_time_s * fs))
    n_frames = int(round(duration_s * fs))

    traces = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=start_frame + n_frames,
        return_scaled=False,
    )

    # shape check: SpikeInterface returns (samples, channels)
    if traces.ndim != 2 or traces.shape[1] != n_channels:
        raise ValueError("Expected traces with shape (n_samples, n_channels).")

    # Demean once, vectorized across all channels
    traces = np.asarray(traces, dtype=np.float32)
    traces -= traces.mean(axis=0, keepdims=True)

    # Welch parameters chosen for speed + frequency resolution
    if nperseg is None:
        # Good default: roughly 1-2 s windows, clipped to practical bounds
        nperseg = int(min(max(fs, 1024), 8192))

    freqs, pxx = welch(
        traces,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        scaling="density",
        axis=0,
    )  # pxx shape: (n_freqs, n_channels)

    # Restrict to band of interest
    fmin, fmax = freq_band
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    f_band = freqs[band_mask]
    p_band = pxx[band_mask]

    if f_band.size < 5:
        raise ValueError("freq_band is too narrow or incompatible with nperseg/fs.")

    # Robust per-channel baseline in band
    band_median = np.median(p_band, axis=0) + 1e-20
    band_max = np.max(p_band, axis=0)
    band_argmax = np.argmax(p_band, axis=0)
    peak_freq = f_band[band_argmax]
    peak_ratio = band_max / band_median

    # Peak prominence, loop only over channels for peak finding
    prominence_ratio = np.zeros(n_channels, dtype=np.float32)
    peak_height = np.zeros(n_channels, dtype=np.float32)

    for ch in range(n_channels):
        y = p_band[:, ch]
        peaks, props = find_peaks(y, prominence=0)
        if peaks.size == 0:
            continue
        best = np.argmax(props["prominences"])
        peak_height[ch] = y[peaks[best]]
        prominence_ratio[ch] = props["prominences"][best] / band_median[ch]

    psd_flag = (peak_ratio >= peak_ratio_threshold) & (prominence_ratio >= peak_prominence_ratio)

    # Optional ACF confirmation
    if require_acf:
        max_lag_samples = int(round(acf_max_period_s * fs))
        min_lag_samples = max(1, int(round(acf_min_period_s * fs)))

        acf = _autocorr_fft(traces, max_lag_samples=max_lag_samples)
        acf_band = acf[min_lag_samples:max_lag_samples + 1]

        acf_peak_val = np.max(acf_band, axis=0)
        acf_peak_idx = np.argmax(acf_band, axis=0) + min_lag_samples
        acf_peak_period_s = acf_peak_idx / fs

        acf_flag = acf_peak_val >= acf_threshold
        final_flag = psd_flag & acf_flag
    else:
        acf = None
        acf_peak_val = None
        acf_peak_period_s = None
        final_flag = psd_flag

    flagged_idx = np.flatnonzero(final_flag)

    # Rank flagged channels by spectral strength, then by ACF if available
    if flagged_idx.size > 0:
        if require_acf:
            rank_score = peak_ratio[flagged_idx] * (1.0 + acf_peak_val[flagged_idx])
        else:
            rank_score = peak_ratio[flagged_idx]
        order = np.argsort(rank_score)[::-1]
        flagged_idx = flagged_idx[order]

    flagged_channel_ids = channel_ids[flagged_idx]

    # -------------------------
    # Plot flagged channels
    # -------------------------
    if flagged_idx.size > 0:
        flagged_idx_plot = flagged_idx[:max_plots]
        raw_len = min(int(round(plot_raw_seconds * fs)), traces.shape[0])
        t_raw = np.arange(raw_len) / fs

        n_show = flagged_idx_plot.size
        fig, axes = plt.subplots(n_show, 3 if require_acf else 2, figsize=(14, 3 * n_show), squeeze=False)

        for row, ch in enumerate(flagged_idx_plot):
            # Raw trace
            ax = axes[row, 0]
            ax.plot(t_raw, traces[:raw_len, ch], lw=0.8)
            ax.set_title(f"Channel {channel_ids[ch]} | raw")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

            # PSD
            ax = axes[row, 1]
            ax.semilogy(f_band, p_band[:, ch], lw=1.0)
            ax.axvline(peak_freq[ch], color="r", ls="--", lw=1)
            title = f"PSD peak={peak_freq[ch]:.2f} Hz | ratio={peak_ratio[ch]:.1f}"
            if prominence_ratio[ch] > 0:
                title += f" | prom={prominence_ratio[ch]:.1f}"
            ax.set_title(title)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")

            # ACF
            if require_acf:
                ax = axes[row, 2]
                lags_s = np.arange(acf.shape[0]) / fs
                ax.plot(lags_s[1:], acf[1:, ch], lw=1.0)
                ax.axhline(acf_threshold, color="r", ls="--", lw=1)
                ax.axvline(acf_peak_period_s[ch], color="g", ls="--", lw=1)
                ax.set_xlim(0, acf_max_period_s)
                ax.set_title(
                    f"ACF peak={acf_peak_val[ch]:.2f} at {acf_peak_period_s[ch]*1000:.1f} ms"
                )
                ax.set_xlabel("Lag (s)")
                ax.set_ylabel("Autocorr")

        plt.tight_layout()
        plt.show()
    else:
        print("No channels passed the repetitive-structure criteria.")

    return {
        "flagged_channel_ids": flagged_channel_ids,
        "flagged_channel_indices": flagged_idx,
        "peak_freq_hz": peak_freq,
        "peak_ratio": peak_ratio,
        "prominence_ratio": prominence_ratio,
        "psd_flag": psd_flag,
        "final_flag": final_flag,
        "freqs_hz": freqs,
        "pxx": pxx,
        "acf": acf,
        "acf_peak_val": acf_peak_val,
        "acf_peak_period_s": acf_peak_period_s,
    }