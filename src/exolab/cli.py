import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from exolab.anomalies import inject_drift, inject_dropout, inject_spike
from exolab.reporting import CaseFile, print_signal_summary, render_case_file
from exolab.signals import generate_signal

app = typer.Typer(name="exolab", add_completion=False, no_args_is_help=True)
console = Console()

_INJECTORS = {
    "spike": inject_spike,
    "dropout": inject_dropout,
    "drift": inject_drift,
}


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
    data = dict(np.load(signal, allow_pickle=True))
    arr = data["signal"]

    anomaly_list: list[dict] = []
    for k, v in data.items():
        if k.startswith("anomaly_"):
            try:
                anomaly_list.append(json.loads(str(v)))
            except (json.JSONDecodeError, ValueError):
                pass

    case = CaseFile(
        signal_path=str(signal),
        length=len(arr),
        freq=float(data["freq"]) if "freq" in data else None,
        noise=float(data["noise"]) if "noise" in data else None,
        anomalies=anomaly_list,
    )

    if case_file:
        render_case_file(case, arr)
    else:
        print_signal_summary(case, arr)
