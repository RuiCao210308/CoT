#!/usr/bin/env bash
set -euo pipefail

TRAIN_JSONL=/root/autodl-tmp/action_chunks_gt_waypoint_trainval_100scenes_train.jsonl
TEST_JSONL=/root/autodl-tmp/action_chunks_gt_waypoint_trainval_100scenes_test.jsonl
DATAROOT=/root/autodl-tmp/data/nuscenes
MODEL_PATH=qwen
FUSION_CKPT=/root/autodl-tmp/openemma_oft_overnight_100scenes_4090_e3/fusion_e3/fusion_action_head.pt
CACHE_ROOT=/root/autodl-tmp/qwen_hidden_cache_trainval_100scenes
SWEEP_ROOT=/root/autodl-tmp/graft_cache_sweep_100scenes

TRAIN_CACHE="${CACHE_ROOT}/train_cache.pt"
TEST_CACHE="${CACHE_ROOT}/test_cache.pt"
SUMMARY_CSV="${SWEEP_ROOT}/summary.csv"
SUMMARY_TXT="${SWEEP_ROOT}/summary.txt"

mkdir -p "${CACHE_ROOT}" "${SWEEP_ROOT}"

if [[ ! -s "${TRAIN_CACHE}" ]]; then
  python3 scripts/build_qwen_hidden_cache.py --jsonl "${TRAIN_JSONL}" --model-path "${MODEL_PATH}" --dataroot "${DATAROOT}" --output_cache "${TRAIN_CACHE}" --device cuda
fi

if [[ ! -s "${TEST_CACHE}" ]]; then
  python3 scripts/build_qwen_hidden_cache.py --jsonl "${TEST_JSONL}" --model-path "${MODEL_PATH}" --dataroot "${DATAROOT}" --output_cache "${TEST_CACHE}" --device cuda
fi

echo "seed,epochs,lr,residual_scale,geometry_sequence_weight,speed_mae_mps,curvature_mae_x100,overall_l1_train_scale,geometry_sequence_ade,geometry_sequence_fde,residual_curvature_abs_mean,checkpoint,eval_json" > "${SUMMARY_CSV}"

for seed in 0 1 2; do
  for residual_scale in 0.05 0.1 0.2 0.3; do
    for geometry_weight in 0.3 0.5; do
      epochs=5
      lr=1e-4
      run_name="seed${seed}_rs${residual_scale}_gw${geometry_weight}_lr${lr}_e${epochs}"
      output_dir="${SWEEP_ROOT}/${run_name}"
      checkpoint="${output_dir}/frozen_fusion_curvature_residual_head.pt"
      eval_json="${output_dir}/eval.json"
      eval_jsonl="${output_dir}/eval_records.jsonl"
      mkdir -p "${output_dir}"

      python3 scripts/train_frozen_fusion_curvature_residual_head.py --jsonl "${TRAIN_JSONL}" --model-path "${MODEL_PATH}" --dataroot "${DATAROOT}" --fusion_checkpoint "${FUSION_CKPT}" --output_dir "${output_dir}" --epochs "${epochs}" --lr "${lr}" --max_samples 1657 --batch_size 1 --device cuda --geometry_sequence_weight "${geometry_weight}" --residual_scale "${residual_scale}" --dt 0.5 --hidden_cache "${TRAIN_CACHE}" --seed "${seed}" > "${output_dir}/train.log" 2>&1

      python3 scripts/eval_action_heads.py --jsonl "${TEST_JSONL}" --model_type frozen_fusion_curvature_residual --checkpoint "${checkpoint}" --model-path "${MODEL_PATH}" --dataroot "${DATAROOT}" --start_index 0 --max_samples 0 --device cuda --output_json "${eval_json}" --output_jsonl "${eval_jsonl}" --hidden_cache "${TEST_CACHE}" > "${output_dir}/eval.log" 2>&1

      python3 - "${eval_json}" "${SUMMARY_CSV}" "${seed}" "${epochs}" "${lr}" "${residual_scale}" "${geometry_weight}" "${checkpoint}" <<'PY'
import csv
import json
import sys

eval_json, summary_csv, seed, epochs, lr, residual_scale, geometry_weight, checkpoint = sys.argv[1:]
with open(eval_json, "r", encoding="utf-8") as f:
    metrics = json.load(f)
row = {
    "seed": seed,
    "epochs": epochs,
    "lr": lr,
    "residual_scale": residual_scale,
    "geometry_sequence_weight": geometry_weight,
    "speed_mae_mps": metrics.get("speed_mae_mps"),
    "curvature_mae_x100": metrics.get("curvature_mae_x100"),
    "overall_l1_train_scale": metrics.get("overall_l1_train_scale"),
    "geometry_sequence_ade": metrics.get("geometry_sequence_ade"),
    "geometry_sequence_fde": metrics.get("geometry_sequence_fde"),
    "residual_curvature_abs_mean": metrics.get("residual_curvature_abs_mean"),
    "checkpoint": checkpoint,
    "eval_json": eval_json,
}
columns = [
    "seed",
    "epochs",
    "lr",
    "residual_scale",
    "geometry_sequence_weight",
    "speed_mae_mps",
    "curvature_mae_x100",
    "overall_l1_train_scale",
    "geometry_sequence_ade",
    "geometry_sequence_fde",
    "residual_curvature_abs_mean",
    "checkpoint",
    "eval_json",
]
with open(summary_csv, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writerow(row)
PY
    done
  done
done

python3 - "${SUMMARY_CSV}" "${SUMMARY_TXT}" <<'PY'
import csv
import sys

summary_csv, summary_txt = sys.argv[1:]
with open(summary_csv, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
columns = [
    "seed",
    "epochs",
    "lr",
    "residual_scale",
    "geometry_sequence_weight",
    "speed_mae_mps",
    "curvature_mae_x100",
    "overall_l1_train_scale",
    "geometry_sequence_ade",
    "geometry_sequence_fde",
    "residual_curvature_abs_mean",
    "checkpoint",
    "eval_json",
]
widths = {column: len(column) for column in columns}
for row in rows:
    for column in columns:
        widths[column] = max(widths[column], len(str(row.get(column, ""))))
lines = ["  ".join(column.ljust(widths[column]) for column in columns)]
lines.append("  ".join("-" * widths[column] for column in columns))
for row in rows:
    lines.append("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"[GraftCacheSweep] wrote {summary_csv}")
print(f"[GraftCacheSweep] wrote {summary_txt}")
PY
