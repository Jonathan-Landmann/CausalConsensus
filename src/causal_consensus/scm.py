"""Synthetic structural causal models with controlled distribution shifts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import random_dag, topological_order


@dataclass(frozen=True)
class SCMConfig:
    n_nodes: int = 20
    n_samples: int = 500
    expected_degree: float = 2.0
    mechanism: str = "linear"
    noise: str = "gaussian"
    shift_strength: float = 0.0
    hidden_confounders: int = 0
    missing_rate: float = 0.0
    seed: int = 7


@dataclass(frozen=True)
class SCMData:
    samples: np.ndarray
    adjacency: np.ndarray
    order: tuple[int, ...]
    coefficients: np.ndarray


def _noise(shape: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(size=shape)
    if kind == "laplace":
        return rng.laplace(scale=1 / np.sqrt(2), size=shape)
    if kind == "student":
        return rng.standard_t(df=3, size=shape) / np.sqrt(3)
    raise ValueError(f"unknown noise distribution: {kind}")


def _mechanism(parent_signal: np.ndarray, kind: str) -> np.ndarray:
    if kind == "linear":
        return parent_signal
    if kind == "tanh":
        return 2.0 * np.tanh(parent_signal)
    if kind == "mixed":
        return 0.65 * parent_signal + 0.7 * np.sin(parent_signal)
    raise ValueError(f"unknown mechanism: {kind}")


def generate_scm(config: SCMConfig) -> SCMData:
    """Generate observational data with optional mechanism and covariate shift."""
    if not 0 <= config.missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1)")
    rng = np.random.default_rng(config.seed)
    adjacency, order = random_dag(config.n_nodes, config.expected_degree, rng)
    coefficients = np.zeros_like(adjacency, dtype=float)
    edge_count = int(adjacency.sum())
    magnitudes = rng.uniform(0.5, 1.8, size=edge_count)
    signs = rng.choice([-1.0, 1.0], size=edge_count)
    coefficients[adjacency.astype(bool)] = magnitudes * signs

    samples = np.zeros((config.n_samples, config.n_nodes), dtype=float)
    environments = rng.integers(0, 2, size=config.n_samples)
    for node in order:
        parents = np.flatnonzero(adjacency[:, node])
        signal = (
            samples[:, parents] @ coefficients[parents, node]
            if len(parents)
            else np.zeros(config.n_samples)
        )
        shifted_signal = signal * (1 + config.shift_strength * environments)
        noise_scale = 1 + 0.5 * config.shift_strength * environments
        samples[:, node] = _mechanism(shifted_signal, config.mechanism) + noise_scale * _noise(
            config.n_samples, config.noise, rng
        )

    for _ in range(config.hidden_confounders):
        children = rng.choice(config.n_nodes, size=min(3, config.n_nodes), replace=False)
        latent = rng.normal(size=config.n_samples)
        samples[:, children] += latent[:, None] * rng.uniform(0.4, 1.0, size=len(children))

    if config.missing_rate:
        mask = rng.random(samples.shape) < config.missing_rate
        samples[mask] = np.nan
    return SCMData(samples, adjacency, tuple(order), coefficients)


def generate_from_graph(
    adjacency: np.ndarray,
    coefficients: np.ndarray,
    n_samples: int,
    mechanism: str = "linear",
    noise: str = "gaussian",
    shift_strength: float = 0.0,
    seed: int = 7,
) -> np.ndarray:
    """Generate a new environment from a fixed causal graph."""
    rng = np.random.default_rng(seed)
    order = topological_order(adjacency)
    samples = np.zeros((n_samples, adjacency.shape[0]), dtype=float)
    for node in order:
        parents = np.flatnonzero(adjacency[:, node])
        signal = samples[:, parents] @ coefficients[parents, node] if len(parents) else 0.0
        samples[:, node] = _mechanism(np.asarray(signal), mechanism) * (1 + shift_strength) + (
            1 + shift_strength / 2
        ) * _noise(n_samples, noise, rng)
    return samples

