"""
Cell-cell graph construction and subgraph sampling.

The graph is a mutual/standard KNN graph built in PCA space (cheap, robust to
noise/dropout). Subgraph sampling (random-walk induced) is used during
contrastive pretraining so that memory/compute scale sub-linearly with the
number of cells, following the scalability strategy used by graph-contrastive
scRNA-seq methods.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def build_knn_graph(
    X: np.ndarray, n_neighbors: int = 10, pca_dim: int = 50, mutual: bool = True
) -> torch.Tensor:
    """Return a (2, num_edges) edge_index tensor of a KNN cell-cell graph.

    If enforcing mutual (reciprocal) edges leaves any node with zero
    neighbors, that node's original (non-mutual) top-1 nearest neighbor edge
    is added back, so every cell keeps at least one edge and the graph has
    no fully isolated nodes (which would otherwise break k-hop subgraph
    sampling and starve the GCN of any message to propagate)."""
    X = np.asarray(X)
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError(f"expected a non-empty 2D expression matrix, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("expression matrix contains NaN or Inf")
    if not 1 <= n_neighbors < X.shape[0]:
        raise ValueError("n_neighbors must be between 1 and n_cells - 1")
    if pca_dim <= 0:
        raise ValueError("pca_dim must be positive")

    n_components = min(pca_dim, X.shape[1], X.shape[0] - 1)
    Z = PCA(n_components=n_components, random_state=0).fit_transform(X)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Z)
    _, idx = nn.kneighbors(Z)

    src = np.repeat(np.arange(Z.shape[0]), n_neighbors)
    dst = idx[:, 1:].reshape(-1)  # drop self-loop (col 0)

    if mutual:
        pair_set = set(zip(src.tolist(), dst.tolist()))
        keep = [(s, d) in pair_set and (d, s) in pair_set for s, d in zip(src, dst)]
        m_src, m_dst = src[keep], dst[keep]

        connected = set(m_src.tolist()) | set(m_dst.tolist())
        isolated = [i for i in range(Z.shape[0]) if i not in connected]
        if isolated:
            fallback_dst = idx[isolated, 1]  # nearest non-self neighbor
            m_src = np.concatenate([m_src, np.array(isolated)])
            m_dst = np.concatenate([m_dst, fallback_dst])
        src, dst = m_src, m_dst

    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_index = np.unique(edge_index, axis=1)
    return torch.tensor(edge_index, dtype=torch.long)


def build_pyg_data(X: np.ndarray, edge_index: torch.Tensor) -> Data:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, n_edges)")
    x = torch.tensor(X, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)


def sample_subgraph(
    data: Data, batch_size: int, num_hops: int = 2, generator: torch.Generator = None
) -> Data:
    """Random-walk-style induced subgraph sampling for scalable training."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = data.num_nodes
    if n <= 0:
        raise ValueError("cannot sample from an empty graph")
    seeds = torch.randint(0, n, (min(batch_size, n),), generator=generator)
    node_idx, edge_index, mapping, _ = k_hop_subgraph(
        seeds,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=n,
    )
    sub = Data(x=data.x[node_idx], edge_index=edge_index)
    sub.orig_idx = node_idx
    sub.seed_mapping = mapping
    return sub
