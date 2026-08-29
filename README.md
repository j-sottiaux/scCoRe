# scCoRe

scCoRe is a two-stage unsupervised framework for single-cell RNA-sequencing
clustering. Stage A learns a graph-contrastive cell representation and derives
confidence-scored pseudo-labels from multiscale consensus clustering. Stage B
uses this pseudo-supervision to refine the graph encoder before final K-means
clustering.

## Installation

Python 3.12 is recommended. The experiments reported in the manuscript used
Python 3.12.13.

```bash
git clone https://github.com/j-sottiaux/scCoRe.git
cd scCoRe

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

The benchmark datasets are not stored in this repository. Download the required
HDF5 files as described in [`data/README.md`](data/README.md) and place them in
the `data/` directory.

scCoRe accepts cells-by-genes HDF5 matrices in either of the following layouts:

- a dense `X` dataset with an optional label vector `Y`;
- a CSR matrix stored under `exprs`, with optional cell labels under
  `obs/cell_type1` or `obs/cell_ontology_class`.

Reference labels are optional and are used only for evaluation.

## Running scCoRe

The command below runs one training seed on the Goolam dataset using the
configuration reported in the manuscript:

```bash
PYTHONHASHSEED=12345 python src/main.py \
    --data_file data/goolam.h5 \
    --num_cluster 5 \
    --random_seed 12345 \
    --device cpu \
    --result_dir results/goolam_seed_12345
```

The defaults implement the manuscript configuration:

| Component | Parameters |
|---|---|
| Preprocessing | genes in at least 3 cells; cells with at least 1 retained gene; library size 10,000; `log1p`; 2,000 Seurat HVGs; scaling clipped at 10 |
| Cell graph | 50-dimensional PCA; 10 reciprocal neighbors; isolated-cell fallback |
| Stage A | GCN `2000→128→32`; 500 epochs; 90 seed cells; 2-hop subgraphs; Adam `1e-3`; 3 adversarial updates of `0.01`; perturbation bound `0.05`; edge dropout `0.2` |
| Consensus | PCA dimensions `5, 10, 15, 20`; 10 repeats per dimension; K-means `n_init=10`; threshold `0.7`; at least `min(5, cluster size)` selected cells per consensus cluster |
| Stage B | 4 attention heads; 2 Transformer layers; feed-forward width 64; dropout `0.1`; 100 epochs; Adam `5e-4`; pseudo-label weight `1.0`; contrastive weight `0.2` |
| Readout | row-wise L2 normalization; K-means `n_init=100`; random states `0, 1, 2` |

The complete command-line options are shown with:

```bash
python src/main.py --help
```

The manuscript experiments used training seeds `1`, `12`, `123`, `1234` and
`12345`. Final embeddings are row-wise L2-normalized and clustered with
K-means using `n_init=100` and random states `0`, `1` and `2`. When changing
the training seed, set `PYTHONHASHSEED` to the same value before starting
Python.

## Outputs

Each run writes:

- `embeddings_and_labels.npz`, containing the aligned cell and feature
  identifiers, available cell metadata, raw and L2-normalized embeddings, and
  the three clustering partitions;
- `metrics.json`, containing per-state and mean ARI, NMI and ACC when reference
  labels are available.

The input HDF5 file is opened in read-only mode and is never modified.

## Tests

Run the lightweight unit and one-epoch workflow tests with:

```bash
python -m unittest discover -s tests -v
```

## Methodological scope

The manuscript preprocessing filters genes detected in fewer than three cells,
removes cells with no retained detected gene, normalizes each cell to 10,000
counts, applies `log1p`, selects 2,000 highly variable genes with Scanpy's
Seurat procedure, and scales values with clipping at 10. Mitochondrial or
ribosomal filtering and batch correction are not performed by this workflow.

## License

The scCoRe source code, including command and code examples in the
documentation, is distributed under the [BSD 3-Clause License](LICENSE). The
textual documentation is distributed under the [Creative Commons Attribution-
NonCommercial 4.0 International License](LICENSE-DOCS.md). Datasets are not
covered by these licenses and remain subject to the terms specified by their
respective providers.

## Acknowledgments

This study was supported by the European Union's Horizon Europe Research and
Innovation Programme under the Marie Skłodowska-Curie Actions (MSCA), Grant
Agreement No. 101236749 (https://thunder-msca-se.univ-lille.fr/#top), and the Initiative of Excellence of the University of
Lille (R-CDP-25-002-PRIME-NEXT-GEN).

## Submission
scCoRe has been submitted to the 2026 IEEE International Conference on Big Data --  Special Session: Healthcare Data
