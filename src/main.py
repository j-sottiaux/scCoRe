"""Public command-line entry point for scCoRe."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# Match the single-threaded CPU protocol reported in the manuscript.
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
import torch

from graph_utils import build_knn_graph, build_pyg_data
from model import HybridModel
from multiscale_consensus import multiscale_consensus
from preprocessing import get_expression_matrix, load_h5, preprocess
from train import finetune_with_consensus, pretrain_contrastive
from utils import evaluate, manuscript_kmeans_readout


MANUSCRIPT_READOUT_STATES = (0, 1, 2)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "scCoRe: confidence-aware consensus-guided representation "
            "learning for scRNA-seq clustering"
        )
    )
    parser.add_argument("--data_file", type=Path, required=True)
    parser.add_argument("--num_cluster", type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=12345)
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--graph_pca_dim", type=int, default=50)
    parser.add_argument("--pca_dims", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--consensus_repeats", type=int, default=10)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--min_pseudo_per_cluster", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--epochs_pretrain", type=int, default=500)
    parser.add_argument("--epochs_finetune", type=int, default=100)
    parser.add_argument("--pseudo_label_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=90)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--result_dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    _validate_cli_arguments(parser, args)
    return args


def _validate_cli_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not args.data_file.is_file():
        parser.error(f"data file not found: {args.data_file}")
    if args.num_cluster < 2:
        parser.error("--num_cluster must be at least 2")
    for name in (
        "n_top_genes",
        "n_neighbors",
        "graph_pca_dim",
        "consensus_repeats",
        "hidden_dim",
        "embed_dim",
        "batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in ("epochs_pretrain", "epochs_finetune"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")
    if not args.pca_dims or any(dimension <= 0 for dimension in args.pca_dims):
        parser.error("--pca_dims must contain positive integers")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence_threshold must be between 0 and 1")
    if args.min_pseudo_per_cluster < 0:
        parser.error("--min_pseudo_per_cluster must be non-negative")
    if args.pseudo_label_weight < 0 or args.contrastive_weight < 0:
        parser.error("loss weights must be non-negative")


def _configure_runtime(seed: int, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    set_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this value only before inter-op work starts.
        pass
    torch.use_deterministic_algorithms(True)


def _validate_preprocessed_data(
    expression: np.ndarray,
    cell_ids: np.ndarray,
    n_clusters: int,
    n_neighbors: int,
) -> None:
    if expression.ndim != 2 or min(expression.shape) == 0:
        raise ValueError(f"expected a non-empty cells-by-genes matrix, got {expression.shape}")
    if not np.isfinite(expression).all():
        raise ValueError("preprocessed expression matrix contains NaN or Inf")
    if expression.shape[0] != cell_ids.size:
        raise ValueError("cell identifiers are not aligned with expression rows")
    if np.unique(cell_ids).size != cell_ids.size:
        raise ValueError("cell identifiers must be unique")
    if n_clusters >= expression.shape[0]:
        raise ValueError("number of clusters must be smaller than number of cells")
    if n_neighbors >= expression.shape[0]:
        raise ValueError("number of neighbors must be smaller than number of cells")


def run(args: argparse.Namespace) -> None:
    """Run one scCoRe training seed and write aligned result artifacts."""
    _configure_runtime(args.random_seed, args.device)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading and preprocessing {args.data_file} ...")
    adata = load_h5(args.data_file)
    adata = preprocess(adata, n_top_genes=args.n_top_genes)
    expression = get_expression_matrix(adata)
    cell_ids = np.asarray(adata.obs_names, dtype=str)
    feature_ids = np.asarray(adata.var_names, dtype=str)
    source_cell_index = adata.obs["source_cell_index"].to_numpy(dtype=np.int64)
    cell_metadata = {
        f"obs_{column}": adata.obs[column].astype(str).to_numpy(dtype=str)
        for column in adata.obs.columns
        if column not in {"cell_id_kind", "source_cell_index"}
    }
    _validate_preprocessed_data(
        expression, cell_ids, args.num_cluster, args.n_neighbors
    )

    print("[2/6] Building the mutual KNN cell graph ...")
    edge_index = build_knn_graph(
        expression,
        n_neighbors=args.n_neighbors,
        pca_dim=args.graph_pca_dim,
    )
    graph_data = build_pyg_data(expression, edge_index)

    print("[3/6] Building multiscale consensus pseudo-labels ...")
    consensus = multiscale_consensus(
        expression,
        n_clusters=args.num_cluster,
        pca_dims=args.pca_dims,
        confidence_threshold=args.confidence_threshold,
        random_state=args.random_seed,
        n_repeats=args.consensus_repeats,
        min_pseudo_per_cluster=args.min_pseudo_per_cluster,
    )
    selected_counts = np.bincount(
        consensus.pseudo_labels[consensus.pseudo_label_mask],
        minlength=args.num_cluster,
    )
    print(
        f"    -> {int(consensus.pseudo_label_mask.sum())}/{expression.shape[0]} "
        f"cells selected; {int((selected_counts > 0).sum())}/{args.num_cluster} "
        "consensus clusters covered"
    )

    print("[4/6] Building the scCoRe model ...")
    model = HybridModel(
        in_dim=expression.shape[1],
        n_clusters=args.num_cluster,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
    )

    print("[5/6] Stage A: graph-contrastive pretraining ...")
    model = pretrain_contrastive(
        model,
        graph_data,
        epochs=args.epochs_pretrain,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("[6/6] Stage B: consensus-guided refinement ...")
    model = finetune_with_consensus(
        model,
        graph_data,
        pseudo_labels=consensus.pseudo_labels,
        confidence=consensus.confidence,
        epochs=args.epochs_finetune,
        pseudo_label_weight=args.pseudo_label_weight,
        contrastive_weight=args.contrastive_weight,
        device=args.device,
    )

    model.eval()
    with torch.no_grad():
        embedding_raw = model.encode(
            graph_data.x.to(args.device), graph_data.edge_index.to(args.device)
        ).cpu().numpy()

    embedding_l2, labels_by_state, _ = manuscript_kmeans_readout(
        embedding_raw,
        args.num_cluster,
    )
    output_path = args.result_dir / "embeddings_and_labels.npz"
    np.savez_compressed(
        output_path,
        cell_ids=cell_ids,
        source_cell_index=source_cell_index,
        feature_ids=feature_ids,
        embedding_raw=embedding_raw,
        embedding_l2=embedding_l2,
        cluster_labels_state_0=labels_by_state[0],
        cluster_labels_state_1=labels_by_state[1],
        cluster_labels_state_2=labels_by_state[2],
        **cell_metadata,
    )
    print(f"Saved embeddings and cluster labels to {output_path}")

    if "true_label" in adata.obs:
        true_labels = adata.obs["true_label"].astype("category").cat.codes.to_numpy()
        metrics_by_state = {
            str(state): evaluate(true_labels, labels_by_state[state])
            for state in MANUSCRIPT_READOUT_STATES
        }
        mean_metrics = {
            metric: float(
                np.mean(
                    [metrics_by_state[str(state)][metric] for state in MANUSCRIPT_READOUT_STATES]
                )
            )
            for metric in ("ARI", "NMI", "ACC")
        }
        metrics_payload = {
            "per_readout_state": metrics_by_state,
            "mean_over_states_0_1_2": mean_metrics,
        }
        metrics_path = args.result_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("Evaluation (mean over states 0, 1 and 2):", mean_metrics)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
