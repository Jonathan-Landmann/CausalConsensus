"""Command-line interface."""

from __future__ import annotations

import argparse
import json

from .experiments import plot_benchmark, run_benchmark
from .pipeline import run_pipeline
from .scm import SCMConfig, generate_scm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-consensus",
        description="Stability-weighted causal subgraph aggregation experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run one synthetic recovery experiment")
    demo.add_argument("--nodes", type=int, default=20)
    demo.add_argument("--samples", type=int, default=500)
    demo.add_argument("--shift", type=float, default=0.5)
    demo.add_argument("--seed", type=int, default=7)
    benchmark = commands.add_parser("benchmark", help="run a multi-seed shift benchmark")
    benchmark.add_argument("--output", default="results")
    benchmark.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        scm = generate_scm(
            SCMConfig(
                n_nodes=args.nodes,
                n_samples=args.samples,
                mechanism="mixed",
                shift_strength=args.shift,
                seed=args.seed,
            )
        )
        result = run_pipeline(scm, subset_size=min(6, args.nodes), seed=args.seed)
        print(json.dumps(result.metrics, indent=2))
    else:
        rows = run_benchmark(args.output, quick=args.quick)
        plot_benchmark(rows, f"{args.output}/shift_benchmark.png")
        print(f"Wrote {len(rows)} benchmark rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

