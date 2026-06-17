import numpy as np 


def within_unit_waveform_corr(waveforms, threshold=0.8, verbose=True):
    mean_waveform = waveforms.mean(axis=0)
    
    wf_centered = waveforms - waveforms.mean(axis=1, keepdims=True)
    mean_centered = mean_waveform - mean_waveform.mean()

    # Pearson correlation, vectorized
    numerator = wf_centered @ mean_centered
    denominator = np.linalg.norm(wf_centered, axis=1) * np.linalg.norm(mean_centered)

    corrs = numerator / denominator

    is_match = corrs >= threshold

    n_match = is_match.sum()
    n_nonmatch = len(is_match) - n_match

    if verbose:
        print(f"Threshold: r >= {threshold}")
        print(f"Matching spikes: {n_match}/{len(is_match)}")
        print(f"Fraction matching: {n_match / len(is_match):.3f}")
    
    return corrs, is_match