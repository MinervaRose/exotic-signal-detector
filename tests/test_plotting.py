import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import numpy as np
import pytest

from exolab.detection import detect
from exolab.plotting import plot_detection


@pytest.fixture
def sig_and_result():
    rng = np.random.default_rng(0)
    sig = np.sin(2 * np.pi * 2.0 * np.linspace(0, 1, 300)) + rng.normal(0, 0.05, 300)
    sig[150] += 10.0
    return sig, detect(sig, window=25, threshold=3.0)


def test_plot_saves_file(sig_and_result, tmp_path):
    sig, result = sig_and_result
    out = tmp_path / "fig.png"
    plot_detection(sig, result, signal_path="test.npz", out=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_runs_without_error(sig_and_result, tmp_path):
    sig, result = sig_and_result
    plot_detection(sig, result, signal_path="test.npz", out=tmp_path / "fig.png")


def test_plot_no_flags_does_not_raise(tmp_path):
    sig = np.ones(100)
    result = detect(sig, window=10, threshold=3.0)
    assert result.flagged_count == 0
    plot_detection(sig, result, out=tmp_path / "fig.png")
