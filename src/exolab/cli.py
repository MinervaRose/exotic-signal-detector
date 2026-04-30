import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from exolab.anomalies import inject_drift, inject_dropout, inject_spike
from exolab.detection import detect as run_detect
from exolab.reporting import CaseFile, print_signal_summary, render_case_file
from exolab.signals import generate_signal

app = typer.Typer(name="exolab", add_completion=False, no_args_is_help=True)
console = Console()

_INJECTORS = {
    "spike": inject_spike,
    "dropout": inject_dropout,
    "drift": inject_drift,
}


def _load_npz(path: Path) -> tuple[np.ndarray, dict]:
    data = dict(np.load(path, allow_pickle=True))
    return data["signal"], data


def _parse_anomalies(data: dict) -> list[dict]:
    result = []
    for k, v in data.items():
        if k.startswith("anomaly_"):
            try:
                result.append(json.loads(str(v)))
            except (json.JSONDecodeError, ValueError):
                pass
    return result


def _build_case(path: Path, arr: np.ndarray, data: dict) -> CaseFile:
    return CaseFile(
        signal_path=str(path),
        length=len(arr),
        freq=float(data["freq"]) if "freq" in data else None,
        noise=float(data["noise"]) if "noise" in data else None,
        anomalies=_parse_anomalies(data),
    )


@app.command()
def generate(
    length: int = typer.Option(1000, help="Number of samples"),
    freq: float = typer.Option(1.0, help="Signal frequency in Hz"),
    noise: float = typer.Option(0.05, help="Gaussian noise standard deviation"),
    out: Path = typer.Option(Path("signal.npz"), help="Output .npz path"),
) -> None:
    """Generate a synthetic sinusoidal time-series signal and save it."""
    data = generate_signal(length=length, freq=freq, noise=noise)
    np.savez(out, **data)
    console.print(f"[green]Saved[/green] {length}-sample signal -> {out}")


@app.command()
def inject(
    signal: Path = typer.Option(..., help="Input .npz signal file"),
    kind: str = typer.Option("spike", help="Anomaly kind: spike | dropout | drift"),
    out: Path = typer.Option(Path("signal_anomaly.npz"), help="Output .npz path"),
) -> None:
    """Inject a synthetic anomaly into a saved signal."""
    if kind not in _INJECTORS:
        console.print(f"[red]Unknown kind '{kind}'. Choose: {', '.join(_INJECTORS)}.[/red]")
        raise typer.Exit(1)

    data = dict(np.load(signal, allow_pickle=True))
    arr = data["signal"]

    modified, meta = _INJECTORS[kind](arr)
    data["signal"] = modified

    anomaly_index = sum(1 for k in data if k.startswith("anomaly_"))
    np.savez(out, **data, **{f"anomaly_{anomaly_index}": np.array(json.dumps(meta))})

    console.print(f"[green]Injected[/green] {kind} anomaly -> {out}")
    console.print(f"  {meta['description']}")


@app.command()
def inspect(
    signal: Path = typer.Option(..., help="Input .npz signal file"),
    case_file: bool = typer.Option(False, "--case-file", help="Render full case file output"),
) -> None:
    """Inspect a saved signal and display a Rich summary."""
    arr, data = _load_npz(signal)
    case = _build_case(signal, arr, data)
    if case_file:
        render_case_file(case, arr)
    else:
        print_signal_summary(case, arr)


@app.command()
def detect(
    signal: Path = typer.Option(..., help="Input .npz signal file"),
    threshold: float = typer.Option(3.0, help="Abs z-score threshold for flagging"),
    window: int = typer.Option(25, help="Rolling mean window size (>= 2)"),
    case_file: bool = typer.Option(False, "--case-file", help="Render full case file output"),
    plot: bool = typer.Option(False, "--plot", help="Show interactive plot"),
    plot_out: Path | None = typer.Option(None, "--plot-out", help="Save plot to file instead of displaying"),
) -> None:
    """Detect anomalies in a saved signal using z-score thresholding."""
    arr, data = _load_npz(signal)
    try:
        result = run_detect(arr, window=window, threshold=threshold)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    case = _build_case(signal, arr, data)
    case.detection = result
    if case_file:
        render_case_file(case, arr)
    else:
        print_signal_summary(case, arr)
    if plot or plot_out is not None:
        from exolab.plotting import plot_detection
        plot_detection(arr, result, signal_path=str(signal), out=plot_out)
