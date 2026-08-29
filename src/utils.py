"""Evaluation metrics and manuscript-aligned clustering helpers."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def l2_normalize_embeddings(embedding: np.ndarray) -> np.ndarray:
    """Return the row-L2 representation used by the manuscript readout."""
    embedding = np.asarray(embedding)
    if embedding.ndim != 2 or min(embedding.shape) == 0:
        raise ValueError(
            f"expected a non-empty 2D embedding, got shape {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains NaN or Inf before L2 normalization")
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embedding contains a zero-norm row")
    normalized = embedding / norms
    if not np.isfinite(normalized).all():
        raise ValueError("embedding contains NaN or Inf after L2 normalization")
    return normalized


def manuscript_kmeans_readout(
    embedding_raw: np.ndarray,
    n_clusters: int,
    random_states: tuple[int, ...] = (0, 1, 2),
    n_init: int = 100,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, dict[str, float | int]]]:
    """Run the fixed scCoRe readout reported in the manuscript."""
    embedding_raw = np.asarray(embedding_raw)
    if embedding_raw.ndim != 2 or min(embedding_raw.shape) == 0:
        raise ValueError("embedding_raw must be a non-empty two-dimensional array")
    if n_clusters < 2 or n_clusters >= embedding_raw.shape[0]:
        raise ValueError("n_clusters must be between 2 and n_cells - 1")
    if tuple(random_states) != (0, 1, 2):
        raise ValueError("manuscript readout states must be exactly (0, 1, 2)")
    if n_init != 100:
        raise ValueError("manuscript K-means n_init must be exactly 100")

    embedding_readout = l2_normalize_embeddings(embedding_raw)
    labels_by_state: dict[int, np.ndarray] = {}
    diagnostics_by_state: dict[int, dict[str, float | int]] = {}
    for state in random_states:
        model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            max_iter=300,
            tol=1e-4,
            random_state=state,
            algorithm="lloyd",
            copy_x=True,
            verbose=0,
        ).fit(embedding_readout)
        labels_by_state[state] = model.labels_.astype(np.int64, copy=True)
        diagnostics_by_state[state] = {
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
        }
    return embedding_readout, labels_by_state, diagnostics_by_state


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Best-match clustering accuracy via the Hungarian algorithm."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.size == 0:
        raise ValueError("cluster labels must be non-empty one-dimensional arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("true and predicted labels must have the same shape")
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("cluster labels must be non-negative integers")
    d = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((d, d), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        cost[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = dict(zip(row_ind, col_ind))
    correct = sum(cost[r, c] for r, c in mapping.items())
    return correct / len(y_true)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("true and predicted labels must be aligned non-empty vectors")
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ACC": float(cluster_accuracy(y_true, y_pred)),
    }
