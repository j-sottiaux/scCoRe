"""
Multi-scale consensus clustering used to generate high-confidence pseudo-labels.

Strategy (inspired by the multi-scale PCA + weighted meta-clustering idea):
  1. Run PCA at several different dimensionalities.
  2. K-means cluster each PCA projection independently.
  3. Build a cell x cell co-association matrix, weighting each base
     clustering by its silhouette quality.
  4. Cluster the consensus co-association matrix (spectral / agglomerative)
     to get a final consensus label per cell.
  5. Keep only cells whose base-clustering votes agree strongly with the
     consensus label (high-confidence subset) as pseudo-labels for the
     downstream Transformer refinement head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


@dataclass
class ConsensusResult:
    consensus_labels: np.ndarray       # (n_cells,) final consensus cluster id
    confidence: np.ndarray             # (n_cells,) fraction of base clusterings agreeing
    pseudo_label_mask: np.ndarray      # (n_cells,) bool, True = high-confidence
    pseudo_labels: np.ndarray          # (n_cells,) consensus label, -1 where not confident


def _select_pseudo_label_mask(
    consensus_labels: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float,
    min_pseudo_per_cluster: int = 0,
) -> np.ndarray:
    """Select confident cells and guarantee minimal consensus-cluster coverage.

    The global confidence threshold remains the primary selection rule. For a
    consensus cluster with fewer than ``min_pseudo_per_cluster`` selected
    cells, the most confident remaining cells from that cluster are added.
    Their original confidence values are preserved for loss weighting.
    """
    if min_pseudo_per_cluster < 0:
        raise ValueError("min_pseudo_per_cluster must be non-negative")
    if consensus_labels.ndim != 1 or confidence.ndim != 1:
        raise ValueError("consensus_labels and confidence must be one-dimensional")
    if consensus_labels.shape != confidence.shape:
        raise ValueError("consensus_labels and confidence must have the same shape")

    mask = confidence >= confidence_threshold
    if min_pseudo_per_cluster == 0:
        return mask

    for cluster in np.unique(consensus_labels):
        cluster_idx = np.flatnonzero(consensus_labels == cluster)
        n_selected = int(mask[cluster_idx].sum())
        n_needed = min(
            min_pseudo_per_cluster - n_selected,
            cluster_idx.size - n_selected,
        )
        if n_needed <= 0:
            continue

        candidates = cluster_idx[~mask[cluster_idx]]
        order = np.lexsort((candidates, -confidence[candidates]))
        mask[candidates[order[:n_needed]]] = True

    return mask


def _base_clusterings(
    X: np.ndarray,
    pca_dims: Sequence[int],
    n_clusters: int,
    random_state: int,
    n_repeats: int = 1,
):
    """Build one base K-means clustering per (PCA dimensionality x repeat).

    Repeating each scale with different K-means seeds turns a single,
    potentially unstable clustering into a small ensemble, which matters a
    lot on small datasets (few cells per cluster) where a single K-means run
    is very sensitive to initialization. Each repeat is weighted by its own
    silhouette score, same as before.
    """
    labels_list, weights = [], []
    for d in pca_dims:
        n_components = min(d, X.shape[1], X.shape[0] - 1)
        Z = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
        for r in range(n_repeats):
            seed = random_state + r
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(Z)
            labels = km.labels_
            try:
                score = silhouette_score(Z, labels, sample_size=min(2000, Z.shape[0]))
            except ValueError:
                score = 0.0
            weights.append(max(score, 1e-3))
            labels_list.append(labels)
    return labels_list, np.array(weights)


def multiscale_consensus(
    X: np.ndarray,
    n_clusters: int,
    pca_dims: Sequence[int] = (5, 10, 15, 20),
    confidence_threshold: float = 0.7,
    random_state: int = 12345,
    n_repeats: int = 10,
    min_pseudo_per_cluster: int = 5,
) -> ConsensusResult:
    X = np.asarray(X)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a finite, non-empty cells-by-features matrix")
    if not 2 <= n_clusters < X.shape[0]:
        raise ValueError("n_clusters must be between 2 and n_cells - 1")
    if not pca_dims or any(dimension <= 0 for dimension in pca_dims):
        raise ValueError("pca_dims must contain positive integers")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    n = X.shape[0]
    labels_list, weights = _base_clusterings(
        X, pca_dims, n_clusters, random_state, n_repeats=n_repeats
    )
    weights = weights / weights.sum()

    # Weighted co-association matrix: how often (and how reliably) two cells
    # land in the same base cluster.
    co_assoc = np.zeros((n, n), dtype=np.float32)
    for labels, w in zip(labels_list, weights):
        same = (labels[:, None] == labels[None, :]).astype(np.float32)
        co_assoc += w * same

    sc_model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
    )
    consensus_labels = sc_model.fit_predict(co_assoc)

    # Voting confidence: fraction of (weighted) base clusterings whose own
    # cluster assignment for a cell is consistent with its consensus cluster
    # membership, measured via average co-association to same-consensus peers.
    confidence = np.zeros(n, dtype=np.float32)
    for c in np.unique(consensus_labels):
        idx = np.where(consensus_labels == c)[0]
        if len(idx) <= 1:
            confidence[idx] = 0.0
            continue
        block = co_assoc[np.ix_(idx, idx)]
        confidence[idx] = (block.sum(axis=1) - np.diag(block)) / (len(idx) - 1)

    mask = _select_pseudo_label_mask(
        consensus_labels,
        confidence,
        confidence_threshold,
        min_pseudo_per_cluster=min_pseudo_per_cluster,
    )
    pseudo_labels = np.where(mask, consensus_labels, -1)

    return ConsensusResult(
        consensus_labels=consensus_labels,
        confidence=confidence,
        pseudo_label_mask=mask,
        pseudo_labels=pseudo_labels,
    )
