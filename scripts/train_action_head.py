#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from action_head import ContinuousActionHead, action_chunk_l1_loss
from qwen_planner import (
    build_qwen_inputs,
    extract_qwen_hidden_states,
    select_planning_hidden,
)


TRAIN_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]
SOURCE_ACTION_SCHEMA = ["speed_mps", "curvature_1pm"]
DEFAULT_QWEN_LOCAL_ID = (
    "/home/Cr_seu0321/.cache/huggingface/hub/"
    "models--Qwen--Qwen2-VL-7B-Instruct/snapshots/"
    "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train only the OpenEMMA OFT-lite continuous action head.")
    parser.add_argument("--jsonl", required=True, help="Path to action_chunks.jsonl.")
    parser.add_argument("--model-path", type=str, default="qwen", help="Qwen local path or qwen alias.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/data/nuscenes")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/openemma_oft_head")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def load_jsonl_records(jsonl_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[ActionHeadTrain] warning: skip line {line_no}, JSON error: {exc}")
                continue
            record["_line_no"] = line_no
            records.append(record)
    return records


def resolve_image_path(record: Dict[str, Any], dataroot: str) -> Optional[str]:
    metadata = record.get("metadata", {})
    image_path = metadata.get("image_path")
    if image_path and os.path.exists(image_path):
        return image_path
    if image_path and not os.path.isabs(image_path):
        candidate = os.path.join(dataroot, image_path)
        if os.path.exists(candidate):
            return candidate
    return None


def get_planning_prompt(record: Dict[str, Any]) -> Optional[str]:
    input_section = record.get("input", {})
    prompt = input_section.get("planning_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return None


def make_train_target(record: Dict[str, Any]) -> Optional[torch.Tensor]:
    target = record.get("target", {}).get("future_action_gt")
    try:
        tensor = torch.as_tensor(target, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tuple(tensor.shape) != (10, 2) or not torch.isfinite(tensor).all():
        return None
    tensor = tensor.clone()
    tensor[:, 1] *= 100.0
    return tensor.unsqueeze(0)


def collect_samples(records: List[Dict[str, Any]], dataroot: str, max_samples: int):
    samples = []
    skipped = Counter()
    for record in records:
        if max_samples > 0 and len(samples) >= max_samples:
            break
        target = make_train_target(record)
        if target is None:
            skipped["invalid_target_future_action_gt"] += 1
            continue
        image_path = resolve_image_path(record, dataroot)
        if image_path is None:
            skipped["missing_image_path"] += 1
            continue
        prompt = get_planning_prompt(record)
        if prompt is None:
            skipped["missing_input_planning_prompt"] += 1
            continue
        samples.append(
            {
                "record": record,
                "image_path": image_path,
                "planning_prompt": prompt,
                "target": target,
            }
        )
    return samples, skipped


def resolve_qwen_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    return os.environ.get("OPENEMMA_QWEN_LOCAL_ID", DEFAULT_QWEN_LOCAL_ID)


def load_qwen_model_and_processor(model_path: str, device: str):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    local_id = resolve_qwen_model_path(model_path)
    print(f"[ActionHeadTrain] loading frozen Qwen from: {local_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        local_id,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        device_map="auto" if "cuda" in device else None,
    )
    processor = AutoProcessor.from_pretrained(local_id, local_files_only=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, processor


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    records = load_jsonl_records(args.jsonl)
    samples, skipped = collect_samples(records, args.dataroot, args.max_samples)
    print(f"[ActionHeadTrain] loaded_records={len(records)} usable_samples={len(samples)} skipped={sum(skipped.values())}")
    if skipped:
        print(f"[ActionHeadTrain] skipped_reasons={dict(skipped)}")
    if not samples:
        print("[ActionHeadTrain] no usable samples. Need metadata.image_path and input.planning_prompt in JSONL.")
        return 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    action_head = ContinuousActionHead(hidden_dim=3584, chunk_size=10, action_dim=2).to(device)
    optimizer = torch.optim.AdamW(action_head.parameters(), lr=args.lr)

    print(f"[ActionHeadTrain] source_action_schema={SOURCE_ACTION_SCHEMA}")
    print(f"[ActionHeadTrain] train_action_schema={TRAIN_ACTION_SCHEMA}")
    print("[ActionHeadTrain] target_train[..., 1] = future_action_gt[..., 1] * 100.0")

    global_step = 0
    accum_count = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_speed_l1 = 0.0
        epoch_curvature_l1 = 0.0
        epoch_samples = 0
        for sample in samples:
            inputs = build_qwen_inputs(
                prompt=sample["planning_prompt"],
                images=sample["image_path"],
                processor=processor,
                model=qwen_model,
                args=args,
            )
            with torch.no_grad():
                last_hidden_state = extract_qwen_hidden_states(inputs, qwen_model)
                planning_hidden = select_planning_hidden(last_hidden_state, inputs["attention_mask"])
            planning_hidden = planning_hidden.to(device=device, dtype=next(action_head.parameters()).dtype)

            pred = action_head(planning_hidden)
            target = sample["target"].to(device=device, dtype=pred.dtype)
            loss = action_chunk_l1_loss(pred, target)
            (loss / max(args.batch_size, 1)).backward()
            accum_count += 1

            speed_l1 = torch.mean(torch.abs(pred[..., 0] - target[..., 0])).detach()
            curvature_l1 = torch.mean(torch.abs(pred[..., 1] - target[..., 1])).detach()
            epoch_loss += float(loss.detach())
            epoch_speed_l1 += float(speed_l1)
            epoch_curvature_l1 += float(curvature_l1)
            epoch_samples += 1

            if accum_count >= args.batch_size:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                global_step += 1
                if global_step % 10 == 0 or global_step == 1:
                    print(
                        f"step={global_step} loss={float(loss.detach()):.6f} "
                        f"speed_l1={float(speed_l1):.6f} curvature_l1={float(curvature_l1):.6f}"
                    )
        if epoch_samples > 0:
            print(
                f"epoch={epoch + 1} avg_loss={epoch_loss / epoch_samples:.6f} "
                f"avg_speed_l1={epoch_speed_l1 / epoch_samples:.6f} "
                f"avg_curvature_l1={epoch_curvature_l1 / epoch_samples:.6f}"
            )

    if accum_count > 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    checkpoint = {
        "action_head_state_dict": action_head.state_dict(),
        "config": {
            "hidden_dim": 3584,
            "chunk_size": 10,
            "action_dim": 2,
            "hidden_size": 1024,
            "dropout": 0.1,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "target_curvature_scale": 100.0,
        },
        "train_action_schema": TRAIN_ACTION_SCHEMA,
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "jsonl_path": args.jsonl,
        "global_step": global_step,
    }
    ckpt_path = os.path.join(args.output_dir, "action_head.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"[ActionHeadTrain] saved checkpoint: {ckpt_path}")
    return 0


def main():
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
