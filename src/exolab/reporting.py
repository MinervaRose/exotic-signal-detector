from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from exolab.detection import DetectionResult

console = Console()

_SPARK_BARS = " ._-~=+*#@"
_MAX_FLAG_DISPLAY = 10


@dataclass
class CaseFile:
    signal_path: str
    length: int
    freq: float | None
    noise: float | None
    anomalies: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    detection: DetectionResult | None = None


def _sparkline(values: np.ndarray, width: int = 48) -> str:
    mn, mx = values.min(), values.max()
    idxs = np.linspace(0, len(values) - 1, width, dtype=int)
    if mx == mn:
        return _SPARK_BARS[0] * width
    normalized = (values[idxs] - mn) / (mx - mn)
    return "".join(_SPARK_BARS[int(v * (len(_SPARK_BARS) - 1))] for v in normalized)


def print_signal_summary(case: CaseFile, signal: np.ndarray) -> None:
    table = Table(title="Signal Summary", box=box.SIMPLE_HEAVY, show_header=True)
    table.add_column("Property", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Path", case.signal_path)
    table.add_row("Samples", str(case.length))
    table.add_row("Mean", f"{signal.mean():.4f}")
    table.add_row("Std", f"{signal.std():.4f}")
    table.add_row("Min", f"{signal.min():.4f}")
    table.add_row("Max", f"{signal.max():.4f}")
    if case.freq is not None:
        table.add_row("Freq (Hz)", f"{case.freq:.3f}")
    if case.noise is not None:
        table.add_row("Noise sigma", f"{case.noise:.4f}")

    console.print(table)
    console.print(Panel(_sparkline(signal), title="Waveform", border_style="dim"))

    if case.anomalies:
        console.rule("[yellow]Injected Anomalies (ALL SYNTHETIC)[/yellow]")
        for a in case.anomalies:
            console.print(f"  [bold]{a['kind'].upper()}[/bold]: {a['description']}")

    if case.notes:
        console.rule("[dim]Notes[/dim]")
        for note in case.notes:
            console.print(f"  {note}")

    if case.detection is not None:
        _print_detection_section(case.detection)


def _print_detection_section(result: DetectionResult) -> None:
    console.rule("[cyan]Detection Results (algorithmic -- z-score thresholding)[/cyan]")

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("Property", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Window", str(result.window))
    table.add_row("Threshold", f"|z| > {result.threshold:.2f}")
    table.add_row("Flagged", str(result.flagged_count))
    if result.flagged_count > 0:
        max_z = float(np.abs(result.z_scores).max())
        table.add_row("Max |z|", f"{max_z:.2f}")
    console.print(table)

    if result.flagged_count > 0:
        indices = np.where(result.flags)[0].tolist()
        shown = indices[:_MAX_FLAG_DISPLAY]
        suffix = f" ... (+{len(indices) - _MAX_FLAG_DISPLAY} more)" if len(indices) > _MAX_FLAG_DISPLAY else ""
        console.print(f"  Flagged indices: {shown}{suffix}")

    console.print(Panel(result.disclaimer, border_style="dim", title="Note"))


def render_case_file(case: CaseFile, signal: np.ndarray) -> None:
    """Full structured case file output. Extended version of print_signal_summary."""
    console.rule("[bold green]EXOLAB CASE FILE[/bold green]")
    print_signal_summary(case, signal)
    console.rule("[dim green]END OF CASE FILE[/dim green]")
