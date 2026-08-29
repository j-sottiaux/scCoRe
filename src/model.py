"""
scCoRe model components combining a graph contrastive encoder with an adversarial
augmenter (graph-side idea) and a self-attention Transformer refinement head
driven by consensus pseudo-labels (multi-scale-consensus-side idea).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge


class GNNEncoder(nn.Module):
    """Two-layer GCN encoder producing cell embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = self.conv2(h, edge_index)
        return h


class ProjectionHead(nn.Module):
    """MLP projection head used only for the contrastive loss (SimCLR-style)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class AdversarialAugmenter(nn.Module):
    """
    Generates a "hard" augmented view of a graph by perturbing node features
    with a learned, gradient-driven adversarial noise vector, plus a
    stochastic edge-drop perturbation of the structure. The noise is trained
    to *maximize* the encoder's contrastive loss (adversary), while the
    encoder is trained to be robust to it (min-max game), which yields more
    informative positive pairs than purely random augmentation.
    """

    def __init__(self, feat_dim: int, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = epsilon
        self.delta = nn.Parameter(torch.zeros(1, feat_dim))

    def perturb_features(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.clamp(self.delta, -self.epsilon, self.epsilon)
        return x + noise

    @staticmethod
    def perturb_structure(edge_index: torch.Tensor, p: float = 0.2) -> torch.Tensor:
        new_edge_index, _ = dropout_edge(edge_index, p=p)
        return new_edge_index

    def reset(self):
        with torch.no_grad():
            self.delta.zero_()


class TransformerRefiner(nn.Module):
    """
    Self-attention refinement head. Treats each mini-batch of cell embeddings
    as a token sequence so attention can model dependencies between cells
    (analogous to using a Transformer over cells in expression space), then
    classifies against the consensus pseudo-labels.
    """

    def __init__(self, in_dim: int, n_clusters: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, in_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=n_heads, dim_feedforward=in_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(in_dim, n_clusters)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(z).unsqueeze(0)          # (1, N, D) - one "sequence" per batch
        h = self.transformer(h).squeeze(0)            # (N, D)
        logits = self.classifier(h)
        return h, logits


class HybridModel(nn.Module):
    """Bundles encoder + augmenter + projection head + transformer refiner."""

    def __init__(self, in_dim: int, n_clusters: int, hidden_dim: int = 128, embed_dim: int = 32):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, embed_dim)
        self.augmenter = AdversarialAugmenter(in_dim)
        self.proj_head = ProjectionHead(embed_dim)
        self.refiner = TransformerRefiner(embed_dim, n_clusters)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)
