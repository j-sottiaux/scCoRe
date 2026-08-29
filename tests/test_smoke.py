"""One-epoch technical smoke test for the complete scCoRe workflow."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_utils import build_knn_graph, build_pyg_data  # noqa: E402
from model import HybridModel  # noqa: E402
from multiscale_consensus import multiscale_consensus  # noqa: E402
from train import finetune_with_consensus, pretrain_contrastive  # noqa: E402
from utils import manuscript_kmeans_readout  # noqa: E402


class WorkflowSmokeTest(unittest.TestCase):
    def test_one_epoch_per_training_stage(self) -> None:
        rng = np.random.default_rng(7)
        expression = np.vstack(
            [
                rng.normal(-1.0, 0.3, size=(10, 12)),
                rng.normal(1.0, 0.3, size=(10, 12)),
            ]
        ).astype(np.float32)

        np.random.seed(7)
        torch.manual_seed(7)
        edge_index = build_knn_graph(expression, n_neighbors=3, pca_dim=5)
        graph_data = build_pyg_data(expression, edge_index)
        consensus = multiscale_consensus(
            expression,
            n_clusters=2,
            pca_dims=(2, 3),
            random_state=7,
            n_repeats=1,
            min_pseudo_per_cluster=2,
        )
        model = HybridModel(
            in_dim=expression.shape[1],
            n_clusters=2,
            hidden_dim=8,
            embed_dim=8,
        )
        model = pretrain_contrastive(
            model,
            graph_data,
            epochs=1,
            batch_size=8,
            device="cpu",
        )
        model = finetune_with_consensus(
            model,
            graph_data,
            consensus.pseudo_labels,
            consensus.confidence,
            epochs=1,
            device="cpu",
        )

        model.eval()
        with torch.no_grad():
            embedding_raw = model.encode(graph_data.x, graph_data.edge_index).numpy()
        _, labels_by_state, _ = manuscript_kmeans_readout(embedding_raw, 2)

        self.assertEqual(embedding_raw.shape, (20, 8))
        self.assertTrue(np.isfinite(embedding_raw).all())
        self.assertEqual(tuple(labels_by_state), (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
