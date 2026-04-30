import numpy as np
import pytest

from exolab.detection import DetectionResult, compute_z_scores, detect, rolling_mean


# --- rolling_mean ---

def test_rolling_mean_shape():
    sig = np.ones(100)
    assert rolling_mean(sig, window=10).shape == (100,)


def test_rolling_mean_known_values():
    # window=3: expanding for i<2, full window for i>=2
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rolling_mean(sig, window=3)
    expected = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(result, expected)


def test_rolling_mean_full_window_accuracy():
    rng = np.random.default_rng(1)
    sig = rng.normal(0, 1, 200)
    window = 20
    rm = rolling_mean(sig, window)
    i = 50
    expected = sig[i - window + 1 : i + 1].mean()
    assert rm[i] == pytest.approx(expected)


# --- compute_z_scores ---

def test_z_scores_shape():
    residuals = np.array([1.0, -1.0, 0.5, -0.5])
    assert compute_z_scores(residuals).shape == (4,)


def test_z_scores_constant_signal():
    result = compute_z_scores(np.zeros(50))
    assert np.all(result == 0.0)


def test_z_scores_known():
    # mean=0, std=1 -> z == residuals
    residuals = np.array([1.0, -1.0, 1.0, -1.0])
    np.testing.assert_allclose(compute_z_scores(residuals), residuals, atol=1e-10)


# --- detect ---

def test_detect_flags_injected_spike():
    rng = np.random.default_rng(7)
    sig = np.sin(2 * np.pi * 3.0 * np.linspace(0, 1, 500)) + rng.normal(0, 0.05, 500)
    sig[250] += 20.0
    result = detect(sig, window=25, threshold=3.0)
    assert result.flags[250]


def test_detect_clean_signal_no_flags():
    # constant signal -> zero residuals -> zero z-scores -> no flags
    result = detect(np.ones(200), window=25, threshold=3.0)
    assert result.flagged_count == 0


def test_detect_result_shapes_consistent():
    sig = np.random.default_rng(3).normal(0, 1, 300)
    r = detect(sig)
    n = len(sig)
    assert r.flags.shape == (n,)
    assert r.z_scores.shape == (n,)
    assert r.residuals.shape == (n,)
    assert r.rolling_mean.shape == (n,)


def test_detect_flagged_count_matches_mask():
    sig = np.random.default_rng(4).normal(0, 1, 200)
    r = detect(sig)
    assert r.flagged_count == int(r.flags.sum())


def test_detect_disclaimer_present():
    r = detect(np.ones(50), window=5, threshold=1.0)
    assert r.disclaimer


# --- input validation ---

def test_validate_window_too_small():
    with pytest.raises(ValueError, match="window"):
        detect(np.ones(50), window=1)


def test_validate_threshold_nonpositive():
    with pytest.raises(ValueError, match="threshold"):
        detect(np.ones(50), threshold=0.0)


def test_validate_signal_not_1d():
    with pytest.raises(ValueError, match="1D"):
        detect(np.ones((10, 10)))


def test_validate_signal_empty():
    with pytest.raises(ValueError, match="non-empty"):
        detect(np.array([]))
