"""Reproducible distribution-shift benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .pipeline import run_pipeline
from .scm import SCMConfig, generate_scm


def run_benchmark(
    output_directory: str | Path,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    shift_strengths: tuple[float, ...] = (0.0, 0.5, 1.0),
    n_nodes: int = 20,
    n_samples: int = 500,
    quick: bool = False,
) -> list[dict]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for shift in shift_strengths:
        for seed in seeds:
            config = SCMConfig(
                n_nodes=n_nodes,
                n_samples=n_samples,
                mechanism="mixed" if shift else "linear",
                noise="student" if shift >= 1 else "gaussian",
                shift_strength=shift,
                seed=seed,
            )
            scm = generate_scm(config)
            result = run_pipeline(
                scm,
                subset_size=min(6, n_nodes),
                n_subsets=20 if quick else 80,
                n_bootstrap=8 if quick else 30,
                seed=seed,
            )
            for method, values in result.metrics.items():
                rows.append(
                    {
                        "seed": seed,
                        "shift_strength": shift,
                        "method": method,
                        **values,
                    }
                )
    if not rows:
        return rows
    fields = sorted({key for row in rows for key in row})
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, dict[str, float]] = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            metric: float(np.mean([row[metric] for row in method_rows if metric in row]))
            for metric in ("f1", "shd", "brier", "ece")
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows


def plot_benchmark(rows: list[dict], output: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install causal-consensus[experiments] to plot results") from error
    methods = sorted({row["method"] for row in rows})
    shifts = sorted({row["shift_strength"] for row in rows})
    colors = {"weighted": "#0f766e", "unweighted": "#f59e0b", "full": "#64748b"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in methods:
        for axis, metric, label in (
            (axes[0], "f1", "Directed F1 (higher is better)"),
            (axes[1], "shd", "Structural Hamming distance (lower is better)"),
        ):
            means = []
            errors = []
            for shift in shifts:
                values = [
                    row[metric]
                    for row in rows
                    if row["method"] == method and row["shift_strength"] == shift
                ]
                means.append(np.mean(values))
                errors.append(np.std(values) / np.sqrt(len(values)))
            axis.errorbar(
                shifts,
                means,
                yerr=errors,
                marker="o",
                linewidth=2,
                capsize=3,
                label=method,
                color=colors.get(method),
            )
            axis.set_xlabel("Distribution-shift strength")
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
