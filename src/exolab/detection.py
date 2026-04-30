from dataclasses import dataclass

import numpy as np

DETECTION_DISCLAIMER = (
    "Flagged points are algorithmic outputs based on z-score thresholding. "
    "They are not proof of real anomalies or physical events."
)


@dataclass
class DetectionResult:
    flags: np.ndarray
    z_scores: np.ndarray
    residuals: np.ndarray
    rolling_mean: np.ndarray
    threshold: float
    window: int
    flagged_count: int
    disclaimer: str


def rolling_mean(signal: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with an expanding window for the first (window-1) samples."""
    s = signal.astype(float)
    n = len(s)
    padded = np.zeros(n + 1)
    padded[1:] = np.cumsum(s)
    expanding = padded[1:window] / np.arange(1, window)
    full = (padded[window:] - padded[: n - window + 1]) / window
    return np.concatenate([expanding, full])


def compute_z_scores(residuals: np.ndarray) -> np.ndarray:
    std = residuals.std()
    if std == 0.0:
        return np.zeros(len(residuals), dtype=float)
    return (residuals - residuals.mean()) / std


def _validate(signal: np.ndarray, window: int, threshold: float) -> None:
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")
    if len(signal) == 0:
        raise ValueError("signal must be non-empty")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")


def detect(
    signal: np.ndarray,
    window: int = 25,
    threshold: float = 3.0,
) -> DetectionResult:
    """Flag samples whose residual z-score exceeds threshold."""
    _validate(signal, window, threshold)
    rm = rolling_mean(signal, window)
    residuals = signal.astype(float) - rm
    z = compute_z_scores(residuals)
    flags = np.abs(z) > threshold
    return DetectionResult(
        flags=flags,
        z_scores=z,
        residuals=residuals,
        rolling_mean=rm,
        threshold=threshold,
        window=window,
        flagged_count=int(flags.sum()),
        disclaimer=DETECTION_DISCLAIMER,
    )
