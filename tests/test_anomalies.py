import numpy as np
import pytest

from exolab.anomalies import (
    SYNTHETIC_DISCLAIMER,
    inject_drift,
    inject_dropout,
    inject_spike,
)


@pytest.fixture
def sig() -> np.ndarray:
    return np.random.default_rng(0).normal(0, 1, 300)


# --- spike ---

def test_spike_changes_exactly_one_sample(sig):
    modified, _ = inject_spike(sig, index=50, magnitude=10.0)
    assert np.count_nonzero(modified != sig) == 1


def test_spike_correct_index(sig):
    modified, meta = inject_spike(sig, index=50, magnitude=10.0)
    assert meta["index"] == 50
    assert modified[50] == pytest.approx(sig[50] + 10.0)


def test_spike_disclaimer(sig):
    _, meta = inject_spike(sig)
    assert SYNTHETIC_DISCLAIMER in meta["description"]


def test_spike_preserves_shape(sig):
    modified, _ = inject_spike(sig)
    assert modified.shape == sig.shape


# --- dropout ---

def test_dropout_zeroes_range(sig):
    modified, _ = inject_dropout(sig, start=10, length=20)
    assert np.all(modified[10:30] == 0.0)


def test_dropout_leaves_rest_intact(sig):
    modified, _ = inject_dropout(sig, start=10, length=20)
    np.testing.assert_array_equal(modified[:10], sig[:10])
    np.testing.assert_array_equal(modified[30:], sig[30:])


def test_dropout_preserves_shape(sig):
    modified, _ = inject_dropout(sig)
    assert modified.shape == sig.shape


def test_dropout_disclaimer(sig):
    _, meta = inject_dropout(sig)
    assert SYNTHETIC_DISCLAIMER in meta["description"]


# --- drift ---

def test_drift_shifts_tail(sig):
    modified, _ = inject_drift(sig, start=100, slope=0.5)
    assert not np.allclose(modified[100:], sig[100:])


def test_drift_preserves_head(sig):
    modified, _ = inject_drift(sig, start=100, slope=0.5)
    np.testing.assert_array_equal(modified[:100], sig[:100])


def test_drift_disclaimer(sig):
    _, meta = inject_drift(sig)
    assert SYNTHETIC_DISCLAIMER in meta["description"]


def test_drift_preserves_shape(sig):
    modified, _ = inject_drift(sig)
    assert modified.shape == sig.shape
