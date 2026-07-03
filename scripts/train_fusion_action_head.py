#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from action_head import FusionActionHead, action_chunk_l1_loss
from qwen_planner import (
    build_qwen_inputs,
    extract_qwen_hidden_states,
    select_planning_hidden,
)
from train_action_head import cached_planning_hidden, load_hidden_cache, load_qwen_model_and_processor


SOURCE_ACTION_SCHEMA = ["speed_mps", "curvature_1pm"]
TRAIN_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenEMMA OFT-lite fusion action head.")
    parser.add_argument("--jsonl", type=str, default="/root/autodl-tmp/action_chunks_gt_stable.jsonl")
    parser.add_argument("--model-path", type=str, default="qwen", help="Qwen local path or qwen alias.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/data/nuscenes")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/openemma_fusion_head")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=100, help="0 means no limit.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--speed_weight", type=float, default=1.0)
    parser.add_argument("--curvature_weight", type=float, default=1.0)
    parser.add_argument("--hidden_cache", type=str, default=None)
    parser.add_argument("--log_interval_steps", type=int, default=0)
    parser.add_argument("--step_history_jsonl", type=str, default=None)
    parser.add_argument("--tensorboard_logdir", type=str, default=None)
    parser.add_argument(
        "--ablation_mode",
        choices=("full", "ego_only", "vlm_only", "shuffled_vlm", "zero_history"),
        default="full",
        help="Input ablation for attribution experiments; does not change the FusionActionHead architecture.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic ablation shuffling.")
    return parser.parse_args()


def load_jsonl_records(jsonl_path: str):
    records = []
    skipped = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped["json_decode_error"] += 1
                continue
            record["_line_no"] = line_no
            records.append(record)
    return records, skipped


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if "cuda" in device_arg and not torch.cuda.is_available():
        print("[FusionHeadTrain] cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_image_path(record: Dict[str, Any], dataroot: str) -> Optional[str]:
    metadata = record.get("metadata", {})
    image_path = metadata.get("image_path")
    if isinstance(image_path, str) and os.path.exists(image_path):
        return image_path
    if isinstance(image_path, str) and not os.path.isabs(image_path):
        candidate = os.path.join(dataroot, image_path)
        if os.path.exists(candidate):
            return candidate
    return None


def tensor_or_none(value: Any, shape) -> Optional[torch.Tensor]:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tuple(tensor.shape) != tuple(shape):
        return None
    if not torch.isfinite(tensor).all():
        return None
    return tensor


def make_train_target(record: Dict[str, Any]) -> Optional[torch.Tensor]:
    target = tensor_or_none(record.get("target", {}).get("future_action_gt"), (10, 2))
    if target is None:
        return None
    target = target.clone()
    target[:, 1] *= 100.0
    return target.unsqueeze(0)


def get_prompt_fields(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    input_section = record.get("input", {})
    system_message = input_section.get("system_message")
    planning_prompt = input_section.get("planning_prompt")
    if not isinstance(system_message, str) or not system_message.strip():
        return None
    if not isinstance(planning_prompt, str) or not planning_prompt.strip():
        return None
    return {"system_message": system_message, "planning_prompt": planning_prompt}


def collect_samples(records: List[Dict[str, Any]], args, use_hidden_cache: bool = False):
    samples = []
    skipped = Counter()
    for record_index, record in enumerate(records):
        if args.max_samples > 0 and len(samples) >= args.max_samples:
            break

        target = make_train_target(record)
        if target is None:
            skipped["invalid_target_future_action_gt"] += 1
            continue
        ego_history = tensor_or_none(record.get("input", {}).get("ego_history_array"), (10, 3))
        if ego_history is None:
            skipped["invalid_input_ego_history_array"] += 1
            continue
        sample = {
            "record_index": record_index,
            "line_no": record.get("_line_no"),
            "ego_history": ego_history.unsqueeze(0),
            "target": target,
        }
        if not use_hidden_cache:
            prompt_fields = get_prompt_fields(record)
            if prompt_fields is None:
                skipped["missing_input_system_or_planning_prompt"] += 1
                continue
            image_path = resolve_image_path(record, args.dataroot)
            if image_path is None:
                skipped["missing_image_path"] += 1
                continue
            sample["image_path"] = image_path
            sample["system_message"] = prompt_fields["system_message"]
            sample["planning_prompt"] = prompt_fields["planning_prompt"]

        samples.append(sample)
    return samples, skipped


def qwen_train_message_builder(system_message: str):
    def _get_message(prompt, image=None, args=None):
        return [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    return _get_message


def attach_shuffled_hidden_indices(samples: List[Dict[str, Any]], seed: int) -> None:
    record_indices = [int(sample["record_index"]) for sample in samples]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(len(record_indices), generator=generator).tolist()
    for sample, shuffled_pos in zip(samples, permutation):
        sample["hidden_record_index"] = record_indices[int(shuffled_pos)]


def apply_ablation_inputs(
    planning_hidden: torch.Tensor,
    ego_history: torch.Tensor,
    ablation_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if ablation_mode == "ego_only":
        planning_hidden = torch.zeros_like(planning_hidden)
    elif ablation_mode in ("vlm_only", "zero_history"):
        ego_history = torch.zeros_like(ego_history)
    return planning_hidden, ego_history


class StepMetricLogger:
    def __init__(self, log_interval_steps: int, jsonl_path: Optional[str], tensorboard_logdir: Optional[str]):
        self.log_interval_steps = int(log_interval_steps or 0)
        self.jsonl_path = jsonl_path
        self.pending: List[Dict[str, float]] = []
        self.step_history: List[Dict[str, Any]] = []
        self.writer = None
        if self.jsonl_path:
            os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
            if os.path.exists(self.jsonl_path):
                print(f"[FusionHeadTrain] appending step history: {self.jsonl_path}")
            else:
                print(f"[FusionHeadTrain] writing step history: {self.jsonl_path}")
        if tensorboard_logdir:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "TensorBoard logging requires tensorboard. Please run: pip install tensorboard"
                ) from exc
            try:

                self.writer = SummaryWriter(log_dir=tensorboard_logdir)
                print(f"[FusionHeadTrain] writing tensorboard logs: {tensorboard_logdir}")
            except Exception as exc:
                raise RuntimeError(f"failed to initialize TensorBoard writer: {tensorboard_logdir}") from exc

    def add(self, metrics: Dict[str, float], step: int, epoch: int, lr: float) -> None:
        if self.log_interval_steps <= 0:
            return
        self.pending.append(metrics)
        if len(self.pending) >= self.log_interval_steps:
            self.flush(step=step, epoch=epoch, lr=lr)

    def flush(self, step: int, epoch: int, lr: float) -> None:
        if self.log_interval_steps <= 0 or not self.pending:
            return
        keys = sorted({key for row in self.pending for key in row})
        record: Dict[str, Any] = {
            "step": int(step),
            "epoch": int(epoch),
            "window_steps": int(len(self.pending)),
            "lr": float(lr),
        }
        for key in keys:
            values = [float(row[key]) for row in self.pending if key in row]
            if values:
                record[key] = sum(values) / len(values)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()
        if self.writer is not None:
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in {"step", "epoch", "window_steps"}:
                    self.writer.add_scalar(f"train/{key}", float(value), int(step))
            self.writer.flush()
        self.step_history.append(record)
        self.pending = []

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def train(args):
    if args.batch_size != 1:
        raise ValueError("train_fusion_action_head.py currently supports batch_size=1 only.")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(int(args.seed))
    records, skipped = load_jsonl_records(args.jsonl)
    hidden_cache = load_hidden_cache(args.hidden_cache)
    if args.ablation_mode == "shuffled_vlm" and hidden_cache is None:
        raise ValueError("--ablation_mode shuffled_vlm requires --hidden_cache for fixed sample-level shuffling.")
    samples, sample_skipped = collect_samples(records, args, use_hidden_cache=hidden_cache is not None)
    if args.ablation_mode == "shuffled_vlm":
        attach_shuffled_hidden_indices(samples, args.seed)
    skipped.update(sample_skipped)
    print(
        f"[FusionHeadTrain] loaded_records={len(records)} usable_samples={len(samples)} "
        f"skipped={sum(skipped.values())}"
    )
    if skipped:
        print(f"[FusionHeadTrain] skipped_reasons={dict(skipped)}")
    print(f"[FusionHeadTrain] source_action_schema={SOURCE_ACTION_SCHEMA}")
    print(f"[FusionHeadTrain] train_action_schema={TRAIN_ACTION_SCHEMA}")
    print("[FusionHeadTrain] target_train[..., 1] = future_action_gt[..., 1] * 100.0")
    print(
        f"[FusionHeadTrain] loss_weights speed_weight={args.speed_weight} "
        f"curvature_weight={args.curvature_weight}"
    )
    print(f"[FusionHeadTrain] ablation_mode={args.ablation_mode} seed={args.seed}")
    if not samples:
        print("[FusionHeadTrain] no usable samples")
        return 1

    device = resolve_device(args.device)
    if hidden_cache is None:
        qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    else:
        qwen_model, processor = None, None
        print(
            f"[FusionHeadTrain] using_hidden_cache={args.hidden_cache} "
            f"num_cached={hidden_cache.get('num_cached')}"
        )
        print("[FusionHeadTrain] hidden cache mode: Qwen model will not be loaded")
    fusion_head = FusionActionHead(
        qwen_hidden_dim=3584,
        history_steps=10,
        ego_dim=3,
        ego_embed_dim=256,
        qwen_embed_dim=512,
        fusion_hidden_size=1024,
        chunk_size=10,
        action_dim=2,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(fusion_head.parameters(), lr=args.lr)

    global_step = 0
    train_history = []
    step_logger = StepMetricLogger(args.log_interval_steps, args.step_history_jsonl, args.tensorboard_logdir)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_speed_l1 = 0.0
        epoch_curvature_l1 = 0.0
        epoch_steps = 0

        for sample in samples:
            if hidden_cache is None:
                inputs = build_qwen_inputs(
                    prompt=sample["planning_prompt"],
                    images=sample["image_path"],
                    processor=processor,
                    model=qwen_model,
                    args=args,
                    get_message_fn=qwen_train_message_builder(sample["system_message"]),
                )
                with torch.no_grad():
                    last_hidden_state = extract_qwen_hidden_states(inputs, qwen_model)
                    planning_hidden = select_planning_hidden(last_hidden_state, inputs["attention_mask"])
            else:
                hidden_record_index = int(sample.get("hidden_record_index", sample["record_index"]))
                planning_hidden = cached_planning_hidden(hidden_cache, hidden_record_index)
            planning_hidden = planning_hidden.to(device=device, dtype=next(fusion_head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            planning_hidden, ego_history = apply_ablation_inputs(planning_hidden, ego_history, args.ablation_mode)
            target = sample["target"].to(device=device, dtype=planning_hidden.dtype)

            pred = fusion_head(planning_hidden, ego_history)
            loss = action_chunk_l1_loss(
                pred,
                target,
                speed_weight=args.speed_weight,
                curvature_weight=args.curvature_weight,
            )
            speed_l1 = torch.mean(torch.abs(pred[..., 0] - target[..., 0])).detach()
            curvature_l1 = torch.mean(torch.abs(pred[..., 1] - target[..., 1])).detach()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_steps += 1
            epoch_loss += float(loss.detach())
            epoch_speed_l1 += float(speed_l1)
            epoch_curvature_l1 += float(curvature_l1)
            step_logger.add(
                {
                    "loss": float(loss.detach()),
                    "speed_l1": float(speed_l1),
                    "curvature_l1": float(curvature_l1),
                },
                step=global_step,
                epoch=epoch + 1,
                lr=optimizer.param_groups[0]["lr"],
            )

            if global_step % 10 == 0 or global_step == 1:
                print(
                    f"step={global_step} loss={float(loss.detach()):.6f} "
                    f"speed_l1={float(speed_l1):.6f} curvature_l1={float(curvature_l1):.6f}"
                )

        epoch_summary = {
            "epoch": epoch + 1,
            "avg_loss": epoch_loss / max(epoch_steps, 1),
            "avg_speed_l1": epoch_speed_l1 / max(epoch_steps, 1),
            "avg_curvature_l1": epoch_curvature_l1 / max(epoch_steps, 1),
            "steps": epoch_steps,
        }
        train_history.append(epoch_summary)
        print(
            f"epoch={epoch_summary['epoch']} avg_loss={epoch_summary['avg_loss']:.6f} "
            f"avg_speed_l1={epoch_summary['avg_speed_l1']:.6f} "
            f"avg_curvature_l1={epoch_summary['avg_curvature_l1']:.6f} "
            f"steps={epoch_summary['steps']}"
        )

    step_logger.flush(step=global_step, epoch=args.epochs, lr=optimizer.param_groups[0]["lr"])
    step_logger.close()
    checkpoint = {
        "fusion_action_head_state_dict": fusion_head.state_dict(),
        "config": {
            "qwen_hidden_dim": 3584,
            "history_steps": 10,
            "ego_dim": 3,
            "ego_embed_dim": 256,
            "qwen_embed_dim": 512,
            "fusion_hidden_size": 1024,
            "chunk_size": 10,
            "action_dim": 2,
            "dropout": 0.1,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "speed_weight": args.speed_weight,
            "curvature_weight": args.curvature_weight,
            "target_curvature_scale": 100.0,
            "hidden_cache": args.hidden_cache,
            "log_interval_steps": args.log_interval_steps,
            "step_history_jsonl": args.step_history_jsonl,
            "tensorboard_logdir": args.tensorboard_logdir,
            "ablation_mode": args.ablation_mode,
            "seed": args.seed,
        },
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "train_action_schema": TRAIN_ACTION_SCHEMA,
        "jsonl_path": args.jsonl,
        "global_step": global_step,
        "train_history": train_history,
        "step_history": step_logger.step_history,
    }
    ckpt_path = os.path.join(args.output_dir, "fusion_action_head.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"[FusionHeadTrain] saved checkpoint: {ckpt_path}")
    return 0


def main():
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
