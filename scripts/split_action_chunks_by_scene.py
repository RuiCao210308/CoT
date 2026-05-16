#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Split action chunk JSONL by scene name.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--test_output", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene_key", type=str, default="scene_name")
    parser.add_argument("--summary_json", type=str, default=None)
    return parser.parse_args()


def load_records(jsonl_path: str, scene_key: str) -> List[Tuple[int, Dict[str, Any], str]]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to decode JSON on line {line_no}: {exc}") from exc
            scene_name = record.get(scene_key)
            if scene_name is None:
                scene_name = record.get("metadata", {}).get(scene_key)
            if not scene_name:
                raise ValueError(f"missing scene name for key '{scene_key}' on line {line_no}")
            records.append((line_no, record, str(scene_name)))
    return records


def write_records(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def build_summary(
    total_records: int,
    scenes: List[str],
    train_scenes: List[str],
    test_scenes: List[str],
    train_records: int,
    test_records: int,
    train_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    return {
        "total_records": total_records,
        "total_scenes": len(scenes),
        "train_scenes": len(train_scenes),
        "test_scenes": len(test_scenes),
        "train_records": train_records,
        "test_records": test_records,
        "train_ratio_requested": train_ratio,
        "train_ratio_actual_by_scene": len(train_scenes) / max(len(scenes), 1),
        "train_ratio_actual_by_record": train_records / max(total_records, 1),
        "seed": seed,
        "train_scene_names": train_scenes[:20],
        "test_scene_names": test_scenes[:20],
    }


def save_summary(path: str, summary: Dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def split_by_scene(args):
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1.")

    records = load_records(args.jsonl, args.scene_key)
    scene_names = sorted({scene_name for _, _, scene_name in records})
    rng = random.Random(args.seed)
    shuffled_scenes = list(scene_names)
    rng.shuffle(shuffled_scenes)
    train_scene_count = int(len(shuffled_scenes) * args.train_ratio)
    if len(shuffled_scenes) > 1:
        train_scene_count = min(max(train_scene_count, 1), len(shuffled_scenes) - 1)
    train_scenes = set(shuffled_scenes[:train_scene_count])
    test_scenes = set(shuffled_scenes[train_scene_count:])

    train_records = []
    test_records = []
    for _, record, scene_name in records:
        if scene_name in train_scenes:
            train_records.append(record)
        elif scene_name in test_scenes:
            test_records.append(record)
        else:
            raise RuntimeError(f"scene {scene_name} was not assigned to train or test")

    write_records(args.train_output, train_records)
    write_records(args.test_output, test_records)

    train_scene_names = sorted(train_scenes)
    test_scene_names = sorted(test_scenes)
    summary = build_summary(
        total_records=len(records),
        scenes=scene_names,
        train_scenes=train_scene_names,
        test_scenes=test_scene_names,
        train_records=len(train_records),
        test_records=len(test_records),
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    if args.summary_json:
        save_summary(args.summary_json, summary)

    print(f"total_records: {summary['total_records']}")
    print(f"total_scenes: {summary['total_scenes']}")
    print(f"train_scenes: {summary['train_scenes']}")
    print(f"test_scenes: {summary['test_scenes']}")
    print(f"train_records: {summary['train_records']}")
    print(f"test_records: {summary['test_records']}")
    print(f"train_ratio_actual_by_scene: {summary['train_ratio_actual_by_scene']}")
    print(f"train_ratio_actual_by_record: {summary['train_ratio_actual_by_record']}")
    if args.summary_json:
        print(f"summary_json: {args.summary_json}")
    print(f"train_output: {args.train_output}")
    print(f"test_output: {args.test_output}")


def main():
    args = parse_args()
    split_by_scene(args)


if __name__ == "__main__":
    main()
