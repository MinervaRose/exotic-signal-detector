import numpy as np
import pytest

from exolab.signals import generate_signal


def test_output_shape():
    data = generate_signal(length=200)
    assert data["signal"].shape == (200,)
    assert data["t"].shape == (200,)


def test_t_bounds():
    data = generate_signal(length=100)
    assert data["t"][0] == pytest.approx(0.0)
    assert data["t"][-1] == pytest.approx(1.0)


def test_metadata_stored():
    data = generate_signal(freq=3.5, noise=0.1)
    assert data["freq"] == pytest.approx(3.5)
    assert data["noise"] == pytest.approx(0.1)


def test_no_nans():
    data = generate_signal(length=500, noise=0.2)
    assert not np.isnan(data["signal"]).any()


def test_reproducible_with_rng():
    rng = np.random.default_rng(0)
    a = generate_signal(length=100, rng=rng)
    rng = np.random.default_rng(0)
    b = generate_signal(length=100, rng=rng)
    np.testing.assert_array_equal(a["signal"], b["signal"])
