import numpy as np


def generate_signal(
    length: int = 1000,
    freq: float = 1.0,
    noise: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Return a dict with keys: t, signal, freq, noise."""
    rng = rng or np.random.default_rng()
    t = np.linspace(0, 1, length)
    signal = np.sin(2 * np.pi * freq * t) + rng.normal(0, noise, length)
    return {"t": t, "signal": signal, "freq": freq, "noise": noise}
