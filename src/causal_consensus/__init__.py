"""CausalConsensus public API."""

from .aggregation import AggregationResult, aggregate_subgraphs
from .bootstrap import LocalEstimate, estimate_stable_subgraph
from .graph import project_to_dag
from .scm import SCMConfig, generate_scm

__all__ = [
    "AggregationResult",
    "LocalEstimate",
    "SCMConfig",
    "aggregate_subgraphs",
    "estimate_stable_subgraph",
    "generate_scm",
    "project_to_dag",
]

__version__ = "0.1.0"

