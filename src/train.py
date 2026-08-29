"""
Two-stage training procedure.

Stage A - Adversarial graph contrastive pretraining (scAGCL-inspired):
  For each subgraph batch, build two views: (1) the encoder's normal output,
  (2) the encoder's output on a structurally-perturbed (edge-dropped) and
  adversarially feature-perturbed copy. The adversary (`delta`) is updated to
  *increase* the contrastive loss (a few inner gradient-ascent steps), then
  the encoder/projection head are updated to *decrease* it, so the encoder
  learns representations robust to worst-case-ish perturbations.

Stage B - Consensus-guided fine-tuning (scMSCF-inspired):
  The full-graph embeddings are passed through the TransformerRefiner and
  trained with a confidence-weighted cross-entropy against the high-confidence
  multi-scale-consensus pseudo-labels, jointly with a light contrastive
  regularizer so the embedding space doesn't collapse toward the (imperfect)
  pseudo-labels.
"""
from __future__ import annotations

from time import perf_counter_ns

import torch
import torch.nn.functional as F

from graph_utils import sample_subgraph
from losses import nt_xent_loss, pseudo_label_loss
from model import HybridModel


def pretrain_contrastive(
    model: HybridModel,
    data,
    epochs: int = 500,
    batch_size: int = 90,
    adv_steps: int = 3,
    adv_lr: float = 0.01,
    lr: float = 1e-3,
    device: str = "cpu",
    loss_records: list[dict] | None = None,
    diagnostic_records: list[dict] | None = None,
    epoch_timing_records: list[dict] | None = None,
):
    model.to(device)
    opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.proj_head.parameters()), lr=lr
    )

    for epoch in range(epochs):
        epoch_start_ns = perf_counter_ns() if epoch_timing_records is not None else None
        sub = sample_subgraph(data, batch_size)
        x, edge_index = sub.x.to(device), sub.edge_index.to(device)

        # --- adversary: find worst-case feature perturbation for this batch ---
        model.augmenter.reset()
        model.augmenter.to(device)
        for inner_step in range(adv_steps):
            model.augmenter.delta.requires_grad_(True)
            x_adv = model.augmenter.perturb_features(x)
            edge_adv = model.augmenter.perturb_structure(edge_index)

            z1 = model.proj_head(model.encode(x, edge_index))
            z2 = model.proj_head(model.encode(x_adv, edge_adv))
            loss_adv = nt_xent_loss(z1, z2)

            if diagnostic_records is not None:
                diagnostic_records.append(
                    {
                        "stage": "stage_a_training",
                        "epoch": epoch + 1,
                        "inner_step": inner_step + 1,
                        "view": "clean_vs_adversarial",
                        "diagnostic_name": "attack_contrastive_loss",
                        "value": float(loss_adv.detach().cpu()),
                    }
                )

            grad = torch.autograd.grad(loss_adv, model.augmenter.delta, retain_graph=False)[0]
            with torch.no_grad():
                model.augmenter.delta += adv_lr * grad.sign()
                model.augmenter.delta.clamp_(-model.augmenter.epsilon, model.augmenter.epsilon)

        # --- encoder step: minimize contrastive loss against the hard view ---
        opt.zero_grad()
        x_adv = model.augmenter.perturb_features(x).detach()
        edge_adv = model.augmenter.perturb_structure(edge_index)
        z1 = model.proj_head(model.encode(x, edge_index))
        z2 = model.proj_head(model.encode(x_adv, edge_adv))
        loss = nt_xent_loss(z1, z2)
        if loss_records is not None:
            loss_records.append(
                {
                    "stage": "stage_a_training",
                    "epoch": epoch + 1,
                    "optimizer_step": epoch + 1,
                    "loss_name": "contrastive_loss",
                    "value": float(loss.detach().cpu()),
                    "weight": 1.0,
                    "is_optimization_objective": True,
                    "measurement_position": "pre_optimizer_step",
                }
            )
        loss.backward()
        opt.step()
        if epoch_timing_records is not None:
            epoch_timing_records.append(
                {
                    "stage": "stage_a_training",
                    "epoch": epoch + 1,
                    "optimizer_step": epoch + 1,
                    "duration_seconds": (perf_counter_ns() - epoch_start_ns) / 1e9,
                }
            )

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[Stage A][{epoch + 1}/{epochs}] contrastive loss = {loss.item():.4f}")

    return model


def finetune_with_consensus(
    model: HybridModel,
    data,
    pseudo_labels,
    confidence,
    epochs: int = 100,
    lr: float = 5e-4,
    pseudo_label_weight: float = 1.0,
    contrastive_weight: float = 0.2,
    device: str = "cpu",
    loss_records: list[dict] | None = None,
    epoch_timing_records: list[dict] | None = None,
):
    if pseudo_label_weight < 0 or contrastive_weight < 0:
        raise ValueError("Stage B loss weights must be non-negative")

    model.to(device)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    pseudo_labels_t = torch.tensor(pseudo_labels, dtype=torch.long, device=device)
    confidence_t = torch.tensor(confidence, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_start_ns = perf_counter_ns() if epoch_timing_records is not None else None
        opt.zero_grad()
        z = model.encode(x, edge_index)
        h, logits = model.refiner(z)

        ce_loss = pseudo_label_loss(logits, pseudo_labels_t, confidence_t)

        # light contrastive regularization on a random feature dropout view
        x_view = F.dropout(x, p=0.1, training=True)
        z_view = model.encode(x_view, edge_index)
        c_loss = nt_xent_loss(model.proj_head(z), model.proj_head(z_view))

        loss = pseudo_label_weight * ce_loss + contrastive_weight * c_loss
        if loss_records is not None:
            common = {
                "stage": "stage_b_training",
                "epoch": epoch + 1,
                "optimizer_step": epoch + 1,
                "measurement_position": "pre_optimizer_step",
            }
            loss_records.extend(
                [
                    {
                        **common,
                        "loss_name": "pseudo_label_loss",
                        "value": float(ce_loss.detach().cpu()),
                        "weight": float(pseudo_label_weight),
                        "is_optimization_objective": False,
                    },
                    {
                        **common,
                        "loss_name": "contrastive_loss",
                        "value": float(c_loss.detach().cpu()),
                        "weight": float(contrastive_weight),
                        "is_optimization_objective": False,
                    },
                    {
                        **common,
                        "loss_name": "total_loss",
                        "value": float(loss.detach().cpu()),
                        "weight": 1.0,
                        "is_optimization_objective": True,
                    },
                ]
            )
        loss.backward()
        opt.step()
        if epoch_timing_records is not None:
            epoch_timing_records.append(
                {
                    "stage": "stage_b_training",
                    "epoch": epoch + 1,
                    "optimizer_step": epoch + 1,
                    "duration_seconds": (perf_counter_ns() - epoch_start_ns) / 1e9,
                }
            )

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"[Stage B][{epoch + 1}/{epochs}] "
                f"pseudo-label loss = {ce_loss.item():.4f}, "
                f"contrastive = {c_loss.item():.4f}, total = {loss.item():.4f} "
                f"(weights: pseudo={pseudo_label_weight:g}, "
                f"contrastive={contrastive_weight:g})"
            )

    return model
