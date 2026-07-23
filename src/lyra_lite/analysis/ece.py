import numpy as np

def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Expected calibration error (ECE)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.clip(np.digitize(y_score, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        bin_mask = (binids == i)
        if np.any(bin_mask):
            bin_acc = y_true[bin_mask].mean()
            bin_conf = y_score[bin_mask].mean()
            bin_weight = np.sum(bin_mask) / len(y_score)
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return float(ece)
