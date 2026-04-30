# exotic-signal-detector

A Python CLI for generating, corrupting, and inspecting synthetic time-series signals.
CLI entry point: `exolab`.

---

## What exolab does

`exolab` lets you:

- **generate** a clean sinusoidal signal with configurable frequency and noise
- **inject** a synthetic anomaly (spike, dropout, or drift) into a saved signal
- **inspect** a saved signal and view a summary table with a waveform sparkline

Anomaly detection is planned for a future milestone. Everything produced today is
synthetic test data — see below.

---

## Install and sync

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync --dev
```

This creates `.venv`, installs all runtime and dev dependencies, and registers the
`exolab` command.

---

## Usage

### generate

Create a 1000-sample, 2 Hz sinusoidal signal:

```
uv run exolab generate --length 1000 --freq 2.0 --noise 0.05 --out signal.npz
```

| Option | Default | Description |
|--------|---------|-------------|
| `--length` | 1000 | Number of samples |
| `--freq` | 1.0 | Frequency in Hz |
| `--noise` | 0.05 | Gaussian noise standard deviation |
| `--out` | signal.npz | Output file path (.npz) |

### inject

Inject a synthetic anomaly into a saved signal:

```
uv run exolab inject --signal signal.npz --kind spike --out signal_anomaly.npz
```

| Option | Default | Description |
|--------|---------|-------------|
| `--signal` | required | Input .npz file |
| `--kind` | spike | `spike`, `dropout`, or `drift` |
| `--out` | signal_anomaly.npz | Output file path (.npz) |

Anomaly kinds:

- **spike** — single-sample amplitude jump
- **dropout** — consecutive samples zeroed out
- **drift** — linear ramp added from a random onset point

Injections are cumulative: you can pipe the output of one `inject` into the next.

### inspect

Display a summary table and ASCII waveform for a saved signal:

```
uv run exolab inspect --signal signal_anomaly.npz
```

Add `--case-file` for a full structured case file header and footer:

```
uv run exolab inspect --signal signal_anomaly.npz --case-file
```

---

## What "synthetic anomaly" means

All anomalies produced by `exolab inject` are **algorithmically constructed test
artifacts**. They are inserted into mathematically generated signals to exercise
detection pipelines.

They are **not** recordings of real-world events, physical sensor failures, or
measurements of any natural phenomenon. Every anomaly description emitted by the
tool carries an explicit `SYNTHETIC -- not a real physical phenomenon` label to
make this clear at every step.

---

## Day 1 status

| Feature | Status |
|---------|--------|
| Signal generation (sinusoidal + noise) | Done |
| Anomaly injection: spike, dropout, drift | Done |
| Signal inspection (table + sparkline) | Done |
| Case file output (`--case-file`) | Done (stub, ready to extend) |
| Anomaly detection | Planned |
| File formats beyond .npz | Planned |

Run the test suite:

```
uv run pytest -v
```

17 tests, all passing.
