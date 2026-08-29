"""
Preprocessing utilities for scRNA-seq count matrices.

Follows the standard pipeline used across most scRNA-seq deep-clustering
methods: drop unexpressed genes, library-size normalize, log1p transform,
select highly-variable genes, and z-scale. Implemented with scanpy so it
plugs directly into an AnnData-based workflow.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
import scanpy as sc


def _decode(arr: np.ndarray) -> np.ndarray:
    """h5py often returns bytes objects for string datasets; decode to str."""
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr


def load_h5(
    path: str | Path,
    label_key: str = "Y",
    canonical_dataset: str | None = None,
) -> ad.AnnData:
    """Load a (cells x genes) count matrix + optional ground-truth labels
    from an .h5 file. Supports two layouts:

    1. Simple layout (e.g. scAGCL benchmark files): dense/sparse matrix at
       `X`, label vector at `Y` (or `label_key`).
    2. AnnData-style layout (common scanpy exports): a CSR sparse matrix
       stored as a group `exprs` with `data`/`indices`/`indptr`/`shape`
       datasets, and cell-type labels under `obs/cell_type1`
       (falls back to `obs/cell_ontology_class` if the former is absent).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    obs_metadata: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        if "obs" in f:
            for key, value in f["obs"].items():
                if isinstance(value, h5py.Dataset) and value.ndim == 1:
                    obs_metadata[key] = _decode(np.asarray(value))

        if "X" in f:
            X = np.asarray(f["X"], dtype=np.float32)
            y = _decode(np.asarray(f[label_key])) if label_key in f else None
            cell_ids = _decode(np.asarray(f["obs_names"])) if "obs_names" in f else None
            feature_ids = _decode(np.asarray(f["var_names"])) if "var_names" in f else None
            cell_id_kind = "source_obs_name" if cell_ids is not None else "synthetic_row_index"

        elif "exprs" in f:
            grp = f["exprs"]
            shape = tuple(np.asarray(grp["shape"]))
            mat = sp.csr_matrix(
                (np.asarray(grp["data"]), np.asarray(grp["indices"]), np.asarray(grp["indptr"])),
                shape=shape,
            )
            X = np.asarray(mat.todense(), dtype=np.float32)
            cell_ids = _decode(np.asarray(f["obs_names"])) if "obs_names" in f else None
            feature_ids = _decode(np.asarray(f["var_names"])) if "var_names" in f else None
            cell_id_kind = (
                "source_obs_name" if cell_ids is not None else "synthetic_row_index"
            )

            y = None
            for key in ("cell_type1", "cell_ontology_class"):
                if key in obs_metadata:
                    y = obs_metadata[key]
                    break
        else:
            raise KeyError(
                f"Unrecognized .h5 layout in {path}: expected top-level 'X' "
                f"or an 'exprs' group. Found keys: {list(f.keys())}"
            )

    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError(f"expected a non-empty cells-by-genes matrix, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("raw expression matrix contains NaN or Inf")
    if np.any(X < 0):
        raise ValueError("raw expression matrix contains negative values")

    if cell_ids is None:
        dataset_prefix = canonical_dataset or "dataset"
        cell_ids = np.asarray(
            [f"{dataset_prefix}:row:{i:06d}" for i in range(X.shape[0])],
            dtype=str,
        )
    if len(cell_ids) != X.shape[0]:
        raise ValueError(
            f"cell identifier count ({len(cell_ids)}) does not match X rows ({X.shape[0]})"
        )
    if np.unique(cell_ids.astype(str)).size != X.shape[0]:
        raise ValueError("cell identifiers must be unique")

    if feature_ids is None:
        feature_ids = np.asarray([f"feature:{i:06d}" for i in range(X.shape[1])])
    if len(feature_ids) != X.shape[1]:
        raise ValueError(
            f"feature identifier count ({len(feature_ids)}) does not match X columns "
            f"({X.shape[1]})"
        )
    if np.unique(feature_ids.astype(str)).size != X.shape[1]:
        raise ValueError("feature identifiers must be unique")

    adata = ad.AnnData(X=X)
    adata.obs_names = np.asarray(cell_ids, dtype=str)
    adata.var_names = np.asarray(feature_ids, dtype=str)
    adata.obs["cell_id_kind"] = cell_id_kind
    adata.obs["source_cell_index"] = np.arange(X.shape[0], dtype=np.int64)
    for key, values in obs_metadata.items():
        if len(values) != X.shape[0]:
            raise ValueError(
                f"obs/{key} length ({len(values)}) does not match X rows ({X.shape[0]})"
            )
        if key not in adata.obs:
            adata.obs[key] = values
    if y is not None:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"expected a one-dimensional label vector, got {y.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"label count ({len(y)}) does not match X rows ({X.shape[0]})"
            )
        adata.obs["true_label"] = y
    return adata


def preprocess(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    min_cells: int = 3,
    scale_max: float = 10.0,
) -> ad.AnnData:
    """Standard QC + normalization + HVG selection pipeline."""
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("AnnData input must contain cells and features")
    if n_top_genes <= 0 or min_cells <= 0 or scale_max <= 0:
        raise ValueError("preprocessing parameters must be positive")
    if adata.obs_names.has_duplicates or adata.var_names.has_duplicates:
        raise ValueError("cell and feature identifiers must be unique")

    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=1)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("QC filtering removed all cells or all features")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if n_top_genes and adata.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat",
            n_bins=20,
            subset=True,
        )

    sc.pp.scale(adata, max_value=scale_max)
    expression = get_expression_matrix(adata)
    if expression.shape != adata.shape or not np.isfinite(expression).all():
        raise ValueError("preprocessed expression matrix is misaligned or non-finite")
    return adata


def get_expression_matrix(adata: ad.AnnData) -> np.ndarray:
    X = adata.X
    expression = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if expression.ndim != 2 or expression.shape != adata.shape:
        raise ValueError("AnnData expression matrix is not aligned with its dimensions")
    return expression
