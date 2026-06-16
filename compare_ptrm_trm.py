from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from eval_ptrm import (
    empty_metric_totals,
    finalize_metric_totals,
    seed_everything,
    update_metric_totals,
)
from models.recursive_reasoning.ptrm import (
    create_ptrm_dataloader,
    load_ptrm_model,
    run_ptrm_rollout,
    summarize_ptrm_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TRM and PTRM on the same checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/TRM/sudoku-extreme-1k-aug-1000/step_65100")
    parser.add_argument("--data-path", default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ptrm-num-trajectories", type=int, default=100)
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


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        checkpoint_name = Path(args.checkpoint).parent.name
        run_name = (
            f"{checkpoint_name}_k{args.ptrm_num_trajectories}_d{args.depth}"
            f"_sigma{args.noise_std:g}_seed{args.seed}"
        )
        output_dir = Path("outputs") / "ptrm_compare" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_metrics(output_dir: Path, metrics: Dict) -> None:
    with open(output_dir / "metrics.json", "wt") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    rows = [
        {"section": "trm", **metrics["trm"]},
        {"section": "ptrm_best_q", **metrics["ptrm_best_q"]},
        {"section": "ptrm_mode", **metrics["ptrm_mode"]},
        {"section": "delta_best_q_minus_trm", **metrics["delta_best_q_minus_trm"]},
        {"section": "delta_mode_minus_trm", **metrics["delta_mode_minus_trm"]},
    ]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_dir / "metrics.csv", "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def delta_metrics(lhs: Dict[str, float], rhs: Dict[str, float]) -> Dict[str, float]:
    return {
        "token_accuracy_delta": lhs["token_accuracy"] - rhs["token_accuracy"],
        "exact_accuracy_delta": lhs["exact_accuracy"] - rhs["exact_accuracy"],
        "mean_q_delta": lhs["mean_q"] - rhs["mean_q"],
    }


def save_predictions(output_dir: Path, saved: Dict[str, List[torch.Tensor]]) -> None:
    if not saved:
        return
    payload = {k: torch.cat(v, dim=0) for k, v in saved.items()}
    torch.save(payload, output_dir / "predictions.pt")


def maybe_append_preds(
    saved: Dict[str, List[torch.Tensor]],
    trm_outputs,
    ptrm_outputs,
    labels: torch.Tensor,
) -> None:
    saved.setdefault("labels", []).append(labels.cpu())
    saved.setdefault("trm_preds", []).append(trm_outputs.single.preds.cpu())
    saved.setdefault("trm_q_scores", []).append(trm_outputs.single.q_scores.cpu())
    saved.setdefault("ptrm_best_q_preds", []).append(ptrm_outputs.best_q.preds.cpu())
    saved.setdefault("ptrm_best_q_scores", []).append(ptrm_outputs.best_q.q_scores.cpu())
    saved.setdefault("ptrm_mode_preds", []).append(ptrm_outputs.mode.preds.cpu())
    saved.setdefault("ptrm_mode_scores", []).append(ptrm_outputs.mode.q_scores.cpu())


def main() -> None:
    args = parse_args()
    if args.ptrm_num_trajectories < 1:
        raise ValueError("--ptrm-num-trajectories must be >= 1")
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
        num_trajectories=args.ptrm_num_trajectories,
        device=args.device,
    )

    totals = {
        "trm": empty_metric_totals(),
        "ptrm_best_q": empty_metric_totals(),
        "ptrm_mode": empty_metric_totals(),
    }
    saved_preds: Dict[str, List[torch.Tensor]] = {}
    processed = 0
    start_time = time.time()

    print(
        f"Comparing TRM vs PTRM: checkpoint={args.checkpoint}, split={args.split}, "
        f"PTRM_K={args.ptrm_num_trajectories}, depth={args.depth}, noise_std={args.noise_std}, "
        f"max_examples={args.max_examples}"
    )

    for set_name, batch, _global_batch_size in dataset:
        remaining = None if args.max_examples == 0 else args.max_examples - processed
        if remaining is not None and remaining <= 0:
            break

        labels = batch["labels"]
        valid_rows = (labels != -100).any(dim=-1)
        available = int(valid_rows.sum().item())
        keep = min(int(remaining), available) if remaining is not None else available
        if keep <= 0:
            break

        batch = {k: v[:keep] for k, v in batch.items()}
        labels = batch["labels"]

        batch_start = time.time()
        trm_logits, trm_q_scores = run_ptrm_rollout(
            model=model,
            batch=batch,
            num_trajectories=1,
            depth=args.depth,
            noise_std=0.0,
            noise_targets=noise_targets,
        )
        trm_outputs = summarize_ptrm_batch(
            traj_logits=trm_logits,
            traj_q_scores=trm_q_scores,
            labels=labels,
            num_trajectories=1,
        )

        ptrm_logits, ptrm_q_scores = run_ptrm_rollout(
            model=model,
            batch=batch,
            num_trajectories=args.ptrm_num_trajectories,
            depth=args.depth,
            noise_std=args.noise_std,
            noise_targets=noise_targets,
        )
        ptrm_outputs = summarize_ptrm_batch(
            traj_logits=ptrm_logits,
            traj_q_scores=ptrm_q_scores,
            labels=labels,
            num_trajectories=args.ptrm_num_trajectories,
        )

        update_metric_totals(totals["trm"], trm_outputs.single.preds, labels, trm_outputs.single.q_scores)
        update_metric_totals(totals["ptrm_best_q"], ptrm_outputs.best_q.preds, labels, ptrm_outputs.best_q.q_scores)
        update_metric_totals(totals["ptrm_mode"], ptrm_outputs.mode.preds, labels, ptrm_outputs.mode.q_scores)

        if args.save_preds:
            maybe_append_preds(saved_preds, trm_outputs, ptrm_outputs, labels)

        processed += keep
        batch_time = time.time() - batch_start
        trm_metrics = finalize_metric_totals(totals["trm"])
        best_q_metrics = finalize_metric_totals(totals["ptrm_best_q"])
        mode_metrics = finalize_metric_totals(totals["ptrm_mode"])
        print(
            f"{set_name}: processed={processed}, batch_examples={keep}, time={batch_time:.2f}s, "
            f"TRM exact={trm_metrics['exact_accuracy']:.4f}, "
            f"PTRM best-Q exact={best_q_metrics['exact_accuracy']:.4f}, "
            f"PTRM mode exact={mode_metrics['exact_accuracy']:.4f}"
        )

    elapsed = time.time() - start_time
    trm_metrics = finalize_metric_totals(totals["trm"])
    ptrm_best_q_metrics = finalize_metric_totals(totals["ptrm_best_q"])
    ptrm_mode_metrics = finalize_metric_totals(totals["ptrm_mode"])

    metrics = {
        "config": {
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "split": args.split,
            "ptrm_num_trajectories": args.ptrm_num_trajectories,
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
        "trm": trm_metrics,
        "ptrm_best_q": ptrm_best_q_metrics,
        "ptrm_mode": ptrm_mode_metrics,
        "delta_best_q_minus_trm": delta_metrics(ptrm_best_q_metrics, trm_metrics),
        "delta_mode_minus_trm": delta_metrics(ptrm_mode_metrics, trm_metrics),
    }

    write_metrics(output_dir, metrics)
    if args.save_preds:
        save_predictions(output_dir, saved_preds)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

