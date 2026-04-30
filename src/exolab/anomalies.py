import numpy as np

SYNTHETIC_DISCLAIMER = "SYNTHETIC -- not a real physical phenomenon"


def inject_spike(
    signal: np.ndarray,
    index: int | None = None,
    magnitude: float = 5.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    rng = rng or np.random.default_rng()
    out = signal.copy()
    idx = index if index is not None else int(rng.integers(0, len(signal)))
    out[idx] += magnitude
    meta = {
        "kind": "spike",
        "index": idx,
        "magnitude": magnitude,
        "description": (
            f"Point amplitude spike at index {idx} (magnitude {magnitude:+.2f}). "
            f"{SYNTHETIC_DISCLAIMER}."
        ),
    }
    return out, meta


def inject_dropout(
    signal: np.ndarray,
    start: int | None = None,
    length: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    rng = rng or np.random.default_rng()
    out = signal.copy()
    s = start if start is not None else int(rng.integers(0, max(1, len(signal) - length)))
    end = min(s + length, len(signal))
    out[s:end] = 0.0
    meta = {
        "kind": "dropout",
        "start": s,
        "length": end - s,
        "description": (
            f"Signal dropout (zeroed) from index {s} to {end - 1}. "
            f"{SYNTHETIC_DISCLAIMER}."
        ),
    }
    return out, meta


def inject_drift(
    signal: np.ndarray,
    start: int | None = None,
    slope: float = 0.01,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    rng = rng or np.random.default_rng()
    out = signal.copy()
    s = start if start is not None else int(rng.integers(0, max(1, len(signal) // 2)))
    ramp = np.arange(len(signal) - s) * slope
    out[s:] += ramp
    meta = {
        "kind": "drift",
        "start": s,
        "slope": slope,
        "description": (
            f"Linear drift from index {s} with slope {slope:+.4f}/sample. "
            f"{SYNTHETIC_DISCLAIMER}."
        ),
    }
    return out, meta
