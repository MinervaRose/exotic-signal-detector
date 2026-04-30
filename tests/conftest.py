import numpy as np
import pytest


@pytest.fixture
def base_signal() -> np.ndarray:
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 500)
    return np.sin(2 * np.pi * 2.0 * t) + rng.normal(0, 0.05, 500)
