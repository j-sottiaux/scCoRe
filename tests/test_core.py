"""Fast unit tests for the public scCoRe core."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_utils import build_knn_graph  # noqa: E402
from main import parse_args  # noqa: E402
from multiscale_consensus import multiscale_consensus  # noqa: E402
from utils import manuscript_kmeans_readout  # noqa: E402


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(12345)
        first = rng.normal(loc=-2.0, scale=0.2, size=(12, 8))
        second = rng.normal(loc=2.0, scale=0.2, size=(12, 8))
        self.expression = np.vstack([first, second]).astype(np.float32)

    def test_graph_has_no_isolated_cells(self) -> None:
        edge_index = build_knn_graph(self.expression, n_neighbors=3, pca_dim=5)
        connected = np.unique(edge_index.numpy())
        np.testing.assert_array_equal(connected, np.arange(self.expression.shape[0]))

    def test_cli_defaults_match_manuscript(self) -> None:
        argv = [
            "main.py",
            "--data_file",
            str(Path(__file__)),
            "--num_cluster",
            "5",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        observed = {
            "n_top_genes": args.n_top_genes,
            "n_neighbors": args.n_neighbors,
            "graph_pca_dim": args.graph_pca_dim,
            "pca_dims": args.pca_dims,
            "consensus_repeats": args.consensus_repeats,
            "confidence_threshold": args.confidence_threshold,
            "min_pseudo_per_cluster": args.min_pseudo_per_cluster,
            "hidden_dim": args.hidden_dim,
            "embed_dim": args.embed_dim,
            "epochs_pretrain": args.epochs_pretrain,
            "epochs_finetune": args.epochs_finetune,
            "pseudo_label_weight": args.pseudo_label_weight,
            "contrastive_weight": args.contrastive_weight,
            "batch_size": args.batch_size,
            "device": args.device,
        }
        expected = {
            "n_top_genes": 2000,
            "n_neighbors": 10,
            "graph_pca_dim": 50,
            "pca_dims": [5, 10, 15, 20],
            "consensus_repeats": 10,
            "confidence_threshold": 0.7,
            "min_pseudo_per_cluster": 5,
            "hidden_dim": 128,
            "embed_dim": 32,
            "epochs_pretrain": 500,
            "epochs_finetune": 100,
            "pseudo_label_weight": 1.0,
            "contrastive_weight": 0.2,
            "batch_size": 90,
            "device": "cpu",
        }
        self.assertEqual(observed, expected)

    def test_consensus_preserves_dimensions_and_cluster_coverage(self) -> None:
        result = multiscale_consensus(
            self.expression,
            n_clusters=2,
            pca_dims=(2, 3),
            random_state=12345,
            n_repeats=2,
            min_pseudo_per_cluster=2,
        )
        self.assertEqual(result.consensus_labels.shape, (24,))
        self.assertEqual(result.confidence.shape, (24,))
        selected = result.pseudo_labels[result.pseudo_label_mask]
        counts = np.bincount(selected, minlength=2)
        self.assertTrue(np.all(counts >= 2))

    def test_manuscript_readout_uses_three_states(self) -> None:
        embedding, labels_by_state, diagnostics = manuscript_kmeans_readout(
            self.expression,
            n_clusters=2,
        )
        self.assertEqual(embedding.shape, self.expression.shape)
        self.assertEqual(tuple(labels_by_state), (0, 1, 2))
        self.assertEqual(tuple(diagnostics), (0, 1, 2))
        for labels in labels_by_state.values():
            self.assertEqual(labels.shape, (24,))


if __name__ == "__main__":
    unittest.main()
