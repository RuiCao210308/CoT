#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Optional

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from qwen_planner import build_qwen_inputs, extract_qwen_hidden_states, select_planning_hidden
from train_action_head import load_qwen_model_and_processor


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Qwen planning-hidden cache aligned to a JSONL file.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--model-path", type=str, default="qwen")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/data/nuscenes")
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means no record limit after start_index.")
    return parser.parse_args()


EXPECTED_HIDDEN_DIM = 3584


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
                record = {"_json_decode_error": True}
            record["_line_no"] = line_no
            records.append(record)
    return records, skipped


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if "cuda" in device_arg and not torch.cuda.is_available():
        print("[QwenHiddenCache] cuda requested but unavailable; falling back to cpu")
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


def get_prompt_fields(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    input_section = record.get("input", {})
    system_message = input_section.get("system_message")
    planning_prompt = input_section.get("planning_prompt")
    if not isinstance(system_message, str) or not system_message.strip():
        return None
    if not isinstance(planning_prompt, str) or not planning_prompt.strip():
        return None
    return {"system_message": system_message, "planning_prompt": planning_prompt}


def qwen_cache_message_builder(system_message: str):
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


def sample_key(record: Dict[str, Any], record_index: int) -> Dict[str, Any]:
    metadata = record.get("metadata", {})
    return {
        "record_index": record_index,
        "line_no": record.get("_line_no"),
        "scene_name": metadata.get("scene_name"),
        "frame_idx": metadata.get("frame_idx"),
        "sample_token": metadata.get("sample_token"),
    }


def main():
    args = parse_args()
    records, load_skipped = load_jsonl_records(args.jsonl)
    start_index = max(args.start_index, 0)
    selected = records[start_index:]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    device = resolve_device(args.device)
    qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    hidden_rows = []
    cache_records = []
    skipped = Counter(load_skipped)

    for offset, record in enumerate(selected):
        record_index = start_index + offset
        entry = sample_key(record, record_index)
        entry["hidden_index"] = None
        if record.get("_json_decode_error"):
            entry["skip_reason"] = "json_decode_error"
            skipped["json_decode_error_record"] += 1
            cache_records.append(entry)
            continue
        prompt_fields = get_prompt_fields(record)
        if prompt_fields is None:
            entry["skip_reason"] = "missing_input_system_or_planning_prompt"
            skipped["missing_input_system_or_planning_prompt"] += 1
            cache_records.append(entry)
            continue
        image_path = resolve_image_path(record, args.dataroot)
        if image_path is None:
            entry["skip_reason"] = "missing_image_path"
            skipped["missing_image_path"] += 1
            cache_records.append(entry)
            continue
        try:
            inputs = build_qwen_inputs(
                prompt=prompt_fields["planning_prompt"],
                images=image_path,
                processor=processor,
                model=qwen_model,
                args=args,
                get_message_fn=qwen_cache_message_builder(prompt_fields["system_message"]),
            )
            last_hidden_state = extract_qwen_hidden_states(inputs, qwen_model)
            planning_hidden = select_planning_hidden(last_hidden_state, inputs["attention_mask"])
        except Exception as exc:
            entry["skip_reason"] = f"qwen_hidden_error:{type(exc).__name__}"
            skipped[entry["skip_reason"]] += 1
            cache_records.append(entry)
            continue
        hidden_index = len(hidden_rows)
        hidden_rows.append(planning_hidden.detach().float().cpu()[0])
        entry["hidden_index"] = hidden_index
        entry["skip_reason"] = None
        cache_records.append(entry)

    hidden = (
        torch.stack(hidden_rows, dim=0)
        if hidden_rows
        else torch.empty(0, EXPECTED_HIDDEN_DIM, dtype=torch.float32)
    )
    if hidden.ndim != 2 or hidden.shape[1] != EXPECTED_HIDDEN_DIM:
        raise ValueError(f"hidden cache tensor must have shape [N,{EXPECTED_HIDDEN_DIM}], got {tuple(hidden.shape)}")
    record_to_hidden_index = {
        int(entry["record_index"]): int(entry["hidden_index"])
        for entry in cache_records
        if entry.get("hidden_index") is not None
    }
    cache = {
        "format": "openemma_qwen_planning_hidden_cache_v1",
        "jsonl_path": args.jsonl,
        "model_path": args.model_path,
        "dataroot": args.dataroot,
        "start_index": start_index,
        "max_samples": args.max_samples,
        "num_records": len(selected),
        "num_cached": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[1]) if hidden.ndim == 2 and hidden.shape[0] > 0 else None,
        "hidden": hidden,
        "records": cache_records,
        "record_to_hidden_index": record_to_hidden_index,
        "skipped": dict(skipped),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_cache)), exist_ok=True)
    torch.save(cache, args.output_cache)
    print(f"[QwenHiddenCache] saved cache: {args.output_cache}")
    print(f"[QwenHiddenCache] num_cached={cache['num_cached']} num_records={cache['num_records']}")
    print(f"[QwenHiddenCache] hidden_shape={tuple(hidden.shape)}")
    if skipped:
        print(f"[QwenHiddenCache] skipped={dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
