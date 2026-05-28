# exotic-signal-detector

![Tests](https://github.com/MinervaRose/exotic-signal-detector/actions/workflows/tests.yml/badge.svg)

![Python](https://img.shields.io/badge/python-3.12-blue)
![CLI](https://img.shields.io/badge/CLI-Typer-009688)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Built with Claude](https://img.shields.io/badge/built%20with-Claude%20Code-blueviolet)
![Data](https://img.shields.io/badge/data-synthetic%20only-lightgrey)

A Python CLI for generating, modifying, and inspecting synthetic time-series
signals, with a baseline anomaly detector. Entry point: `exolab`.

This project is a deliberate exercise in a structured, test-driven workflow.
All data is synthetic. No real measurements or physical phenomena are involved.

**Pipeline:** generate → inject → detect → plot → inspect

---

## Features

- Generate sinusoidal signals with configurable frequency and noise
- Inject synthetic anomalies: spike, dropout, drift
- Detect anomalies using rolling-mean residual z-score thresholding
- Visualize signal, rolling mean, and flagged points with matplotlib
- Inspect any saved signal with a Rich terminal summary or full case file report
- 35 tests, all passing

---

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync --dev
```

Creates `.venv`, installs all dependencies, and registers the `exolab` command.

---

## Usage

Entry point: `exolab`

### generate — create a signal

```
uv run exolab generate --length 1000 --freq 2.0 --noise 0.05 --out signal.npz
```

| Option | Default | Description |
|--------|---------|-------------|
| `--length` | 1000 | Number of samples |
| `--freq` | 1.0 | Frequency in Hz |
| `--noise` | 0.05 | Gaussian noise standard deviation |
| `--out` | signal.npz | Output file (.npz) |

### inject — add a synthetic anomaly

```
uv run exolab inject --signal signal.npz --kind spike --out signal_spike.npz
```

| Option | Default | Description |
|--------|---------|-------------|
| `--signal` | required | Input .npz file |
| `--kind` | spike | `spike`, `dropout`, or `drift` |
| `--out` | signal_anomaly.npz | Output file (.npz) |

Injections are cumulative — pipe the output of one `inject` into the next.

### detect — flag anomalies

```
uv run exolab detect --signal signal_spike.npz --threshold 3.0 --window 25
```

Show an interactive plot:

```
uv run exolab detect --signal signal_spike.npz --plot
```

Save the plot to a file:

```
uv run exolab detect --signal signal_spike.npz --plot-out detection.png
```

| Option | Default | Description |
|--------|---------|-------------|
| `--signal` | required | Input .npz file |
| `--threshold` | 3.0 | Abs z-score cutoff |
| `--window` | 25 | Rolling mean window size |
| `--plot` | off | Show interactive plot |
| `--plot-out` | None | Save plot to this path |
| `--case-file` | off | Render full case file output |

### inspect — view a signal summary

```
uv run exolab inspect --signal signal_spike.npz
uv run exolab inspect --signal signal_spike.npz --case-file
```

---

## Detection method

`exolab detect` uses a simple, deterministic baseline:

1. Compute a **rolling mean** of the signal (expanding window for the first
   `window - 1` samples, full window thereafter).
2. Compute **residuals**: `signal − rolling_mean`.
3. Compute **z-scores** of the residuals: `(residual − mean) / std`.
4. **Flag** any sample where `|z| > threshold`.

This method is intentionally minimal and explainable. It is sensitive to
local amplitude deviations, not to frequency or phase anomalies. It is a
starting point, not a production detector.

---

## Notes on synthetic data

All signals and anomalies produced by `exolab` are algorithmically constructed.

- **Signals** are mathematical sine waves plus Gaussian noise. They are not
  recordings of any physical system.
- **Anomalies** (spike, dropout, drift) are programmatically inserted
  perturbations. Every anomaly description emitted by the tool carries the
  label `SYNTHETIC -- not a real physical phenomenon`.
- **Detected flags** are the output of a z-score threshold test. A flag means
  a sample is statistically unusual relative to its local neighbourhood under
  this model. It is not evidence of a real event.

---

## Running tests

```
uv run pytest -v
```
