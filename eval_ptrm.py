from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from models.recursive_reasoning.ptrm import (
    create_ptrm_dataloader,
    load_ptrm_model,
    run_ptrm_rollout,
    summarize_ptrm_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PTRM rollouts on Sudoku checkpoints.")
    parser.add_argument("--checkpoint", default="checkpoints/TRM/sudoku-extreme-1k-aug-1000/step_65100")
    parser.add_argument("--data-path", default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--noise-targets", default="z_h,z_l")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=1024, help="0 means full split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-preds", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        checkpoint_name = Path(args.checkpoint).parent.name
        run_name = (
            f"{checkpoint_name}_k{args.num_trajectories}_d{args.depth}"
            f"_sigma{args.noise_std:g}_seed{args.seed}"
        )
        output_dir = Path("outputs") / "ptrm" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def empty_metric_totals() -> Dict[str, float]:
    return {
        "token_correct": 0.0,
        "token_total": 0.0,
        "exact_correct": 0.0,
        "count": 0.0,
        "q_sum": 0.0,
    }


def update_metric_totals(totals: Dict[str, float], preds: torch.Tensor, labels: torch.Tensor, q_scores: torch.Tensor) -> None:
    valid_mask = labels != -100
    valid_examples = valid_mask.any(dim=-1)
    totals["token_correct"] += float(((preds == labels) & valid_mask).sum().item())
    totals["token_total"] += float(valid_mask.sum().item())
    totals["exact_correct"] += float((((preds == labels) | ~valid_mask).all(dim=-1) & valid_examples).sum().item())
    totals["count"] += float(valid_examples.sum().item())
    totals["q_sum"] += float(q_scores.float().sum().item())


def finalize_metric_totals(totals: Dict[str, float]) -> Dict[str, float]:
    token_total = max(totals["token_total"], 1.0)
    count = max(totals["count"], 1.0)
    return {
        "token_accuracy": totals["token_correct"] / token_total,
        "exact_accuracy": totals["exact_correct"] / count,
        "mean_q": totals["q_sum"] / count,
        "count": totals["count"],
    }


def write_metrics(output_dir: Path, metrics: Dict) -> None:
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "wt") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    rows = []
    for name in ("single", "best_q", "mode"):
        row = {"selection": name}
        row.update(metrics[name])
        rows.append(row)

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_append_preds(saved: Dict[str, List[torch.Tensor]], baseline_outputs, outputs, labels: torch.Tensor) -> None:
    saved.setdefault("labels", []).append(labels.cpu())
    saved.setdefault("single_preds", []).append(baseline_outputs.single.preds.cpu())
    saved.setdefault("single_q_scores", []).append(baseline_outputs.single.q_scores.cpu())
    saved.setdefault("best_q_preds", []).append(outputs.best_q.preds.cpu())
    saved.setdefault("mode_preds", []).append(outputs.mode.preds.cpu())
    saved.setdefault("traj_q_scores", []).append(outputs.traj_q_scores.cpu())


def save_predictions(output_dir: Path, saved: Dict[str, List[torch.Tensor]]) -> None:
    if not saved:
        return
    payload = {k: torch.cat(v, dim=0) for k, v in saved.items()}
    torch.save(payload, output_dir / "predictions.pt")


def main() -> None:
    args = parse_args()
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be >= 1")
    if args.depth < 1:
        raise ValueError("--depth must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    seed_everything(args.seed)
    output_dir = make_output_dir(args)
    noise_targets = tuple(x.strip().lower() for x in args.noise_targets.split(",") if x.strip())
    unknown_targets = set(noise_targets) - {"z_h", "z_l"}
    if unknown_targets:
        raise ValueError(f"Unknown --noise-targets values: {sorted(unknown_targets)}")

    dataset, metadata = create_ptrm_dataloader(
        data_path=args.data_path,
        split=args.split,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = load_ptrm_model(
        checkpoint_path=args.checkpoint,
        metadata=metadata,
        batch_size=args.batch_size,
        num_trajectories=args.num_trajectories,
        device=args.device,
    )

    totals = {
        "single": empty_metric_totals(),
        "best_q": empty_metric_totals(),
        "mode": empty_metric_totals(),
    }
    saved_preds: Dict[str, List[torch.Tensor]] = {}
    processed = 0
    start_time = time.time()

    print(
        f"Evaluating PTRM: checkpoint={args.checkpoint}, split={args.split}, "
        f"K={args.num_trajectories}, depth={args.depth}, noise_std={args.noise_std}, "
        f"max_examples={args.max_examples}"
    )

    for set_name, batch, _global_batch_size in dataset:
        remaining = None if args.max_examples == 0 else args.max_examples - processed
        if remaining is not None and remaining <= 0:
            break

        labels = batch["labels"]
        valid_rows = (labels != -100).any(dim=-1)
        available = int(valid_rows.sum().item())
        if remaining is not None:
            keep = min(int(remaining), available)
        else:
            keep = available
        if keep <= 0:
            break
        batch = {k: v[:keep] for k, v in batch.items()}
        labels = batch["labels"]

        batch_start = time.time()
        baseline_logits, baseline_q_scores = run_ptrm_rollout(
            model=model,
            batch=batch,
            num_trajectories=1,
            depth=args.depth,
            noise_std=0.0,
            noise_targets=noise_targets,
        )
        baseline_outputs = summarize_ptrm_batch(
            traj_logits=baseline_logits,
            traj_q_scores=baseline_q_scores,
            labels=labels,
            num_trajectories=1,
        )

        traj_logits, traj_q_scores = run_ptrm_rollout(
            model=model,
            batch=batch,
            num_trajectories=args.num_trajectories,
            depth=args.depth,
            noise_std=args.noise_std,
            noise_targets=noise_targets,
        )
        outputs = summarize_ptrm_batch(
            traj_logits=traj_logits,
            traj_q_scores=traj_q_scores,
            labels=labels,
            num_trajectories=args.num_trajectories,
        )

        update_metric_totals(totals["single"], baseline_outputs.single.preds, labels, baseline_outputs.single.q_scores)
        update_metric_totals(totals["best_q"], outputs.best_q.preds, labels, outputs.best_q.q_scores)
        update_metric_totals(totals["mode"], outputs.mode.preds, labels, outputs.mode.q_scores)

        if args.save_preds:
            maybe_append_preds(saved_preds, baseline_outputs, outputs, labels)

        processed += keep
        batch_time = time.time() - batch_start
        print(
            f"{set_name}: processed={processed}, batch_examples={keep}, "
            f"time={batch_time:.2f}s, best_q_exact={finalize_metric_totals(totals['best_q'])['exact_accuracy']:.4f}, "
            f"mode_exact={finalize_metric_totals(totals['mode'])['exact_accuracy']:.4f}"
        )

    elapsed = time.time() - start_time
    metrics = {
        "config": {
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "split": args.split,
            "num_trajectories": args.num_trajectories,
            "depth": args.depth,
            "noise_std": args.noise_std,
            "noise_targets": list(noise_targets),
            "batch_size": args.batch_size,
            "device": args.device,
            "max_examples": args.max_examples,
            "seed": args.seed,
        },
        "processed_examples": processed,
        "elapsed_seconds": elapsed,
        "examples_per_second": processed / max(elapsed, 1e-9),
        "single": finalize_metric_totals(totals["single"]),
        "best_q": finalize_metric_totals(totals["best_q"]),
        "mode": finalize_metric_totals(totals["mode"]),
    }
    write_metrics(output_dir, metrics)
    if args.save_preds:
        save_predictions(output_dir, saved_preds)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

