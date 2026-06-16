from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
from collections import OrderedDict
import json
import time

import yaml
import torch

from dataset.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class


@dataclass(frozen=True)
class PtrmSelection:
    preds: torch.Tensor
    q_scores: torch.Tensor


@dataclass(frozen=True)
class PtrmBatchOutputs:
    traj_preds: torch.Tensor
    traj_q_scores: torch.Tensor
    single: PtrmSelection
    best_q: PtrmSelection
    mode: PtrmSelection


def load_checkpoint_config(checkpoint_path: str | Path) -> Dict:
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent / "all_config.yaml"
    with open(config_path, "rt") as f:
        return yaml.safe_load(f)


def build_model_config(
    checkpoint_config: Dict,
    metadata: PuzzleDatasetMetadata,
    batch_size: int,
    num_trajectories: int,
) -> Dict:
    arch_cfg = dict(checkpoint_config["arch"])
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    arch_cfg["batch_size"] = batch_size * num_trajectories
    arch_cfg["vocab_size"] = metadata.vocab_size
    arch_cfg["seq_len"] = metadata.seq_len
    arch_cfg["num_puzzle_identifiers"] = metadata.num_puzzle_identifiers
    return arch_cfg


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    normalized = OrderedDict()
    for key, value in state_dict.items():
        while key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        normalized[key] = value
    return normalized


def load_ptrm_model(
    checkpoint_path: str | Path,
    metadata: PuzzleDatasetMetadata,
    batch_size: int,
    num_trajectories: int,
    device: str | torch.device,
) -> TinyRecursiveReasoningModel_ACTV1:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_config = load_checkpoint_config(checkpoint_path)
    model_cfg = build_model_config(checkpoint_config, metadata, batch_size=batch_size, num_trajectories=num_trajectories)
    model_cls = load_model_class(checkpoint_config["arch"]["name"])

    model = model_cls(model_cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = normalize_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model


def create_sudoku_dataset(
    data_path: str | Path,
    split: str,
    batch_size: int,
    seed: int,
) -> tuple[PuzzleDataset, PuzzleDatasetMetadata]:
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=seed,
            dataset_paths=[str(data_path)],
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split=split,
    )
    return dataset, dataset.metadata


def expand_batch(batch: Dict[str, torch.Tensor], num_trajectories: int) -> Dict[str, torch.Tensor]:
    return {k: v.repeat_interleave(num_trajectories, dim=0) for k, v in batch.items()}


def _initial_inner_carry(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch_size: int,
    device: torch.device,
) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
    seq_len = model.inner.config.seq_len + model.inner.puzzle_emb_len
    hidden = model.inner.config.hidden_size
    dtype = model.inner.H_init.dtype
    carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=torch.empty((batch_size, seq_len, hidden), device=device, dtype=dtype),
        z_L=torch.empty((batch_size, seq_len, hidden), device=device, dtype=dtype),
    )
    reset_flag = torch.ones(batch_size, device=device, dtype=torch.bool)
    return model.inner.reset_carry(reset_flag, carry)


def _apply_noise(
    carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    noise_std: float,
    noise_targets: Sequence[str],
) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
    if noise_std <= 0:
        return carry

    target_set = {target.strip().lower() for target in noise_targets}

    def add_noise(tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor, dtype=torch.float32) * noise_std
        return tensor + noise.to(dtype=tensor.dtype)

    z_H = add_noise(carry.z_H) if "z_h" in target_set else carry.z_H
    z_L = add_noise(carry.z_L) if "z_l" in target_set else carry.z_L
    return TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)


def run_ptrm_rollout(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    num_trajectories: int,
    depth: int,
    noise_std: float,
    noise_targets: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    traj_batch = expand_batch({k: v.to(device) for k, v in batch.items()}, num_trajectories)

    with torch.inference_mode():
        carry = _initial_inner_carry(model, traj_batch["inputs"].shape[0], device=device)
        logits = None
        q_halt_logits = None
        for _ in range(depth):
            carry = _apply_noise(carry, noise_std=noise_std, noise_targets=noise_targets)
            carry, logits, (q_halt_logits, _) = model.inner(carry, traj_batch)

    assert logits is not None and q_halt_logits is not None
    return logits, q_halt_logits


def _sequence_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    valid_mask = labels != IGNORE_LABEL_ID
    valid_examples = valid_mask.any(dim=-1)
    token_total = valid_mask.sum().item()
    token_correct = ((preds == labels) & valid_mask).sum().item()
    exact_correct = (((preds == labels) | ~valid_mask).all(dim=-1) & valid_examples).sum().item()
    valid_count = valid_examples.sum().item()
    return {
        "token_correct": float(token_correct),
        "token_total": float(token_total),
        "exact_correct": float(exact_correct),
        "count": float(valid_count),
    }


def _pick_mode(preds: torch.Tensor, q_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    unique_preds, inverse, counts = torch.unique(preds, dim=0, return_inverse=True, return_counts=True)
    best_count = counts.max()
    candidate_ids = torch.nonzero(counts == best_count, as_tuple=False).flatten()

    if candidate_ids.numel() == 1:
        chosen_uid = candidate_ids[0]
    else:
        best_uid = candidate_ids[0]
        best_q = None
        for uid in candidate_ids.tolist():
            candidate_q = q_scores[inverse == uid].max()
            candidate_q_value = candidate_q.item()
            if best_q is None or candidate_q_value > best_q:
                best_q = candidate_q_value
                best_uid = torch.tensor(uid, device=preds.device)
        chosen_uid = best_uid

    chosen_mask = inverse == chosen_uid
    chosen_indices = torch.nonzero(chosen_mask, as_tuple=False).flatten()
    chosen_local = q_scores[chosen_mask].argmax()
    chosen_idx = chosen_indices[chosen_local]
    return preds[chosen_idx], q_scores[chosen_idx]


def summarize_ptrm_batch(
    traj_logits: torch.Tensor,
    traj_q_scores: torch.Tensor,
    labels: torch.Tensor,
    num_trajectories: int,
) -> PtrmBatchOutputs:
    seq_preds = torch.argmax(traj_logits, dim=-1)
    batch_size = labels.shape[0]
    seq_len = labels.shape[1]

    traj_preds = seq_preds.view(batch_size, num_trajectories, seq_len).detach().cpu()
    traj_q_scores = traj_q_scores.view(batch_size, num_trajectories).detach().cpu()
    labels = labels.detach().cpu()

    single_preds = traj_preds[:, 0]
    single_q = traj_q_scores[:, 0]

    best_q_idx = traj_q_scores.argmax(dim=1)
    best_q_preds = traj_preds[torch.arange(batch_size), best_q_idx]
    best_q_scores = traj_q_scores.max(dim=1).values

    mode_preds = []
    mode_q_scores = []
    for i in range(batch_size):
        pred, q_score = _pick_mode(traj_preds[i], traj_q_scores[i])
        mode_preds.append(pred)
        mode_q_scores.append(q_score)
    mode_preds = torch.stack(mode_preds, dim=0)
    mode_q_scores = torch.stack(mode_q_scores, dim=0)

    return PtrmBatchOutputs(
        traj_preds=traj_preds,
        traj_q_scores=traj_q_scores,
        single=PtrmSelection(preds=single_preds, q_scores=single_q),
        best_q=PtrmSelection(preds=best_q_preds, q_scores=best_q_scores),
        mode=PtrmSelection(preds=mode_preds, q_scores=mode_q_scores),
    )


def aggregate_selection_metrics(selection: PtrmSelection, labels: torch.Tensor) -> Dict[str, float]:
    metrics = _sequence_metrics(selection.preds, labels)
    token_total = max(metrics["token_total"], 1.0)
    count = max(metrics["count"], 1.0)
    return {
        "token_accuracy": metrics["token_correct"] / token_total,
        "exact_accuracy": metrics["exact_correct"] / count,
        "mean_q": selection.q_scores.float().mean().item() if selection.q_scores.numel() else 0.0,
        "count": metrics["count"],
    }


def create_ptrm_dataloader(
    data_path: str | Path,
    split: str,
    batch_size: int,
    seed: int,
) -> tuple[Iterable, PuzzleDatasetMetadata]:
    dataset, metadata = create_sudoku_dataset(data_path=data_path, split=split, batch_size=batch_size, seed=seed)
    return dataset, metadata

