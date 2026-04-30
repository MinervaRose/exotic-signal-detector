from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from exolab.detection import DetectionResult


def plot_detection(
    signal: np.ndarray,
    result: DetectionResult,
    signal_path: str = "",
    out: Path | None = None,
) -> None:
    """Plot signal, rolling mean, and flagged points. Saves to out or shows interactively."""
    fig, ax = plt.subplots()

    x = np.arange(len(signal))

    ax.plot(x, signal, linewidth=0.8, label="Signal")
    ax.plot(x, result.rolling_mean, linewidth=1.2, label=f"Rolling mean (w={result.window})")

    if result.flagged_count > 0:
        flagged_x = np.where(result.flags)[0]
        ax.scatter(flagged_x, signal[flagged_x], color="red", zorder=3,
                   label=f"Flagged (|z| > {result.threshold})")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    title = f"Detection: {signal_path}" if signal_path else "Detection"
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    if out is not None:
        fig.savefig(out)
    else:
        plt.show()

    plt.close(fig)
