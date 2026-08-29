"""Loss functions for the two training stages."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Standard normalized temperature-scaled cross-entropy contrastive loss
    over two augmented views (z1, z2 already L2-normalized)."""
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                       # (2N, D)
    sim = z @ z.T / temperature                           # (2N, 2N)
    sim.fill_diagonal_(-1e9)

    targets = torch.arange(n, device=z.device)
    targets = torch.cat([targets + n, targets], dim=0)    # positive pair indices

    return F.cross_entropy(sim, targets)


def pseudo_label_loss(
    logits: torch.Tensor, pseudo_labels: torch.Tensor, confidence: torch.Tensor
) -> torch.Tensor:
    """Confidence-weighted cross-entropy over cells with a valid pseudo-label
    (pseudo_labels == -1 marks 'unlabeled', excluded)."""
    mask = pseudo_labels >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    ce = F.cross_entropy(logits[mask], pseudo_labels[mask], reduction="none")
    w = confidence[mask]
    return (ce * w).sum() / (w.sum() + 1e-8)
