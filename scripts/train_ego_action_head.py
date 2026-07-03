#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from action_head import EgoOnlyActionHead, action_chunk_l1_loss


SOURCE_ACTION_SCHEMA = ["speed_mps", "curvature_1pm"]
TRAIN_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train ego-only action head baseline.")
    parser.add_argument("--jsonl", type=str, default="/root/autodl-tmp/action_chunks_gt_stable.jsonl")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/openemma_ego_head")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--speed_weight", type=float, default=1.0)
    parser.add_argument("--curvature_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_records(jsonl_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[EgoHeadTrain] skip line {line_no}: {exc}")
                continue
            records.append(record)
    return records


def build_sample(record: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    ego_history = record.get("input", {}).get("ego_history_array")
    target = record.get("target", {}).get("future_action_gt")
    try:
        ego_tensor = torch.as_tensor(ego_history, dtype=torch.float32)
        target_tensor = torch.as_tensor(target, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tuple(ego_tensor.shape) != (10, 3) or tuple(target_tensor.shape) != (10, 2):
        return None
    if not torch.isfinite(ego_tensor).all() or not torch.isfinite(target_tensor).all():
        return None

    target_tensor = target_tensor.clone()
    target_tensor[:, 1] *= 100.0
    return {"ego_history": ego_tensor, "target": target_tensor}


def collect_samples(records: List[Dict[str, Any]], max_samples: int):
    samples = []
    skipped = 0
    for record in records:
        if max_samples > 0 and len(samples) >= max_samples:
            break
        sample = build_sample(record)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)
    return samples, skipped


def batch_iter(samples: List[Dict[str, torch.Tensor]], batch_size: int):
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        ego_history = torch.stack([item["ego_history"] for item in batch], dim=0)
        target = torch.stack([item["target"] for item in batch], dim=0)
        yield ego_history, target


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    records = load_records(args.jsonl)
    samples, skipped = collect_samples(records, args.max_samples)
    print(f"[EgoHeadTrain] loaded_records={len(records)} usable_samples={len(samples)} skipped={skipped}")
    print(f"[EgoHeadTrain] source_action_schema={SOURCE_ACTION_SCHEMA}")
    print(f"[EgoHeadTrain] train_action_schema={TRAIN_ACTION_SCHEMA}")
    print(
        f"[EgoHeadTrain] loss_weights speed_weight={args.speed_weight} "
        f"curvature_weight={args.curvature_weight}"
    )
    if not samples:
        print("[EgoHeadTrain] no usable samples")
        return 1

    head = EgoOnlyActionHead(history_steps=10, ego_dim=3, chunk_size=10, action_dim=2).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for ego_history, target in batch_iter(samples, args.batch_size):
            ego_history = ego_history.to(device)
            target = target.to(device)
            pred = head(ego_history)
            loss = action_chunk_l1_loss(
                pred,
                target,
                speed_weight=args.speed_weight,
                curvature_weight=args.curvature_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_steps += 1
            epoch_loss += float(loss.detach())
            print(f"step={global_step} loss={float(loss.detach()):.6f}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"epoch={epoch + 1} avg_loss={avg_loss:.6f}")

    checkpoint = {
        "ego_action_head_state_dict": head.state_dict(),
        "config": {
            "history_steps": 10,
            "ego_dim": 3,
            "chunk_size": 10,
            "action_dim": 2,
            "hidden_size": 256,
            "dropout": 0.1,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "speed_weight": args.speed_weight,
            "curvature_weight": args.curvature_weight,
            "target_curvature_scale": 100.0,
            "seed": args.seed,
        },
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "train_action_schema": TRAIN_ACTION_SCHEMA,
        "jsonl_path": args.jsonl,
        "global_step": global_step,
    }
    ckpt_path = os.path.join(args.output_dir, "ego_action_head.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"[EgoHeadTrain] saved checkpoint: {ckpt_path}")
    return 0


def main():
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
