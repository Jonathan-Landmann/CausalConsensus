# Preliminary benchmark artifacts

Generated with:

```bash
python -m causal_consensus.cli benchmark --output results/full
```

Configuration: five seeds, shift strengths 0.0/0.5/1.0, 20 nodes, 500 samples, 80 subsets, and 30 bootstrap replicates. `benchmark.csv` contains every run; `summary.json` contains averages across shifts and seeds; `shift_benchmark.png` plots mean performance with standard-error bars.

These artifacts verify reproducibility and expose the current negative result: full-data ordered regression outperforms both subgraph methods. They are not a preregistered paper benchmark.
