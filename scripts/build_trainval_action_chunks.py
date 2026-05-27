#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Set


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Build and scene-split trainval action chunk JSONL files.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp")
    parser.add_argument("--num_scenes", type=int, default=300)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history_steps", type=int, default=10)
    parser.add_argument("--future_steps", type=int, default=10)
    parser.add_argument("--camera", type=str, default="CAM_FRONT")
    parser.add_argument("--require_camera_file", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--max_abs_curvature_1pm", type=float, default=0.2)
    parser.add_argument("--min_segment_distance", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_command(command: List[str]) -> None:
    print("[BuildTrainvalActionChunks] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def scene_names(path: str) -> Set[str]:
    scenes: Set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            scene = record.get("metadata", {}).get("scene_name") or record.get("scene_name")
            if not scene:
                raise ValueError(f"missing scene_name in {path} line {line_no}")
            scenes.add(str(scene))
    return scenes


def record_count(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, value: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def ensure_outputs_are_safe(paths: List[str], overwrite: bool) -> None:
    existing = [path for path in paths if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing outputs; pass --overwrite to replace: "
            + ", ".join(existing)
        )


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"trainval_{args.num_scenes}scenes"
    prefix = os.path.join(args.output_dir, f"action_chunks_gt_waypoint_{tag}")
    all_jsonl = f"{prefix}_all.jsonl"
    train_jsonl = f"{prefix}_train.jsonl"
    test_jsonl = f"{prefix}_test.jsonl"
    split_summary_json = f"{prefix}_split_summary.json"
    all_sanity_json = f"{prefix}_all_sanity.json"
    train_sanity_json = f"{prefix}_train_sanity.json"
    test_sanity_json = f"{prefix}_test_sanity.json"

    outputs = [
        all_jsonl,
        train_jsonl,
        test_jsonl,
        split_summary_json,
        all_sanity_json,
        train_sanity_json,
        test_sanity_json,
    ]
    ensure_outputs_are_safe(outputs, args.overwrite)

    build_cmd = [
        sys.executable,
        "scripts/build_gt_action_chunks.py",
        "--dataroot",
        args.dataroot,
        "--version",
        args.version,
        "--output_jsonl",
        all_jsonl,
        "--history_steps",
        str(args.history_steps),
        "--future_steps",
        str(args.future_steps),
        "--camera",
        args.camera,
        "--num_scenes",
        str(args.num_scenes),
        "--require_camera_file",
        str(args.require_camera_file),
        "--max_abs_curvature_1pm",
        str(args.max_abs_curvature_1pm),
        "--min_segment_distance",
        str(args.min_segment_distance),
    ]
    split_cmd = [
        sys.executable,
        "scripts/split_action_chunks_by_scene.py",
        "--jsonl",
        all_jsonl,
        "--train_output",
        train_jsonl,
        "--test_output",
        test_jsonl,
        "--train_ratio",
        str(args.train_ratio),
        "--seed",
        str(args.seed),
        "--scene_key",
        "scene_name",
        "--summary_json",
        split_summary_json,
    ]
    sanity_jobs = [
        (all_jsonl, all_sanity_json),
        (train_jsonl, train_sanity_json),
        (test_jsonl, test_sanity_json),
    ]

    run_command(build_cmd)
    run_command(split_cmd)
    for jsonl, sanity_json in sanity_jobs:
        run_command(
            [
                sys.executable,
                "openemma/planner/action_chunk_sanity.py",
                jsonl,
                "--report_output",
                sanity_json,
                "--expected_future_len",
                str(args.future_steps),
                "--max_abs_curvature_1pm",
                str(args.max_abs_curvature_1pm),
            ]
        )

    all_scenes = scene_names(all_jsonl)
    train_scenes = scene_names(train_jsonl)
    test_scenes = scene_names(test_jsonl)
    overlap = sorted(train_scenes & test_scenes)
    sanity = {
        "all": load_json(all_sanity_json).get("status"),
        "train": load_json(train_sanity_json).get("status"),
        "test": load_json(test_sanity_json).get("status"),
    }
    summary = load_json(split_summary_json)
    summary.update(
        {
            "all_jsonl": all_jsonl,
            "train_jsonl": train_jsonl,
            "test_jsonl": test_jsonl,
            "all_records": record_count(all_jsonl),
            "train_records": record_count(train_jsonl),
            "test_records": record_count(test_jsonl),
            "all_scenes": len(all_scenes),
            "train_scenes": len(train_scenes),
            "test_scenes": len(test_scenes),
            "scene_overlap_count": len(overlap),
            "scene_overlap_names": overlap,
            "scene_split_disjoint": len(overlap) == 0,
            "sanity_reports": {
                "all": all_sanity_json,
                "train": train_sanity_json,
                "test": test_sanity_json,
            },
            "sanity_status": sanity,
        }
    )
    save_json(split_summary_json, summary)

    print("[BuildTrainvalActionChunks] final summary")
    print(f"  all_records={summary['all_records']} all_scenes={summary['all_scenes']}")
    print(f"  train_records={summary['train_records']} train_scenes={summary['train_scenes']}")
    print(f"  test_records={summary['test_records']} test_scenes={summary['test_scenes']}")
    print(f"  scene_split_disjoint={summary['scene_split_disjoint']}")
    print(f"  sanity_status={summary['sanity_status']}")
    print(f"  split_summary_json={split_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
