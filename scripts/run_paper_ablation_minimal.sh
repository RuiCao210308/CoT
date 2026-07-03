#!/usr/bin/env bash
set -euo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-${AUTODL_ROOT}/data/nuscenes}"
CACHE_ROOT="${CACHE_ROOT:-${AUTODL_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AUTODL_ROOT}/openemma_oft_ablation_minimal}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
TB_ROOT="${TB_ROOT:-${OUTPUT_ROOT}/tensorboard}"
MODEL_PATH="${MODEL_PATH:-qwen}"
DEVICE="${DEVICE:-cuda}"
EPOCHS_300="${EPOCHS_300:-5}"
EPOCHS_850="${EPOCHS_850:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
SEED="${SEED:-0}"
RUN_TINY_CHECKS="${RUN_TINY_CHECKS:-1}"

TRAIN_JSONL_100="${TRAIN_JSONL_100:-${AUTODL_ROOT}/action_chunks_gt_waypoint_trainval_100scenes_train.jsonl}"
TEST_JSONL_100="${TEST_JSONL_100:-${AUTODL_ROOT}/action_chunks_gt_waypoint_trainval_100scenes_test.jsonl}"
TRAIN_JSONL_300="${TRAIN_JSONL_300:-${AUTODL_ROOT}/action_chunks_gt_waypoint_trainval_300scenes_train.jsonl}"
TEST_JSONL_300="${TEST_JSONL_300:-${AUTODL_ROOT}/action_chunks_gt_waypoint_trainval_300scenes_test.jsonl}"
TRAIN_JSONL_850="${TRAIN_JSONL_850:-${AUTODL_ROOT}/action_chunks_gt_waypoint_trainval_850scenes_train.jsonl}"
TEST_JSONL_850="${TEST_JSONL_850:-${AUTODL_ROOT}/action_chunks_gt_waypoint_trainval_850scenes_test.jsonl}"

TRAIN_CACHE_300="${TRAIN_CACHE_300:-${CACHE_ROOT}/qwen_hidden_cache_trainval_300scenes/train_cache.pt}"
TEST_CACHE_300="${TEST_CACHE_300:-${CACHE_ROOT}/qwen_hidden_cache_trainval_300scenes/test_cache.pt}"
TRAIN_CACHE_850="${TRAIN_CACHE_850:-${CACHE_ROOT}/qwen_hidden_cache_trainval_850scenes/train_cache.pt}"
TEST_CACHE_850="${TEST_CACHE_850:-${CACHE_ROOT}/qwen_hidden_cache_trainval_850scenes/test_cache.pt}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TB_ROOT}"
FAILURE_LOG="${OUTPUT_ROOT}/failures.log"
: > "${FAILURE_LOG}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[AblationRunner] missing required file: ${path}" | tee -a "${FAILURE_LOG}"
    exit 1
  fi
}

check_required_paths() {
  if [[ "${BATCH_SIZE}" != "1" ]]; then
    echo "[AblationRunner] BATCH_SIZE=${BATCH_SIZE} is not supported by train_fusion_action_head.py; keep BATCH_SIZE=1 for these ablations." | tee -a "${FAILURE_LOG}"
    exit 1
  fi
  check_file "${TRAIN_JSONL_300}"
  check_file "${TEST_JSONL_300}"
  check_file "${TRAIN_JSONL_850}"
  check_file "${TEST_JSONL_850}"
  check_file "${TRAIN_CACHE_300}"
  check_file "${TEST_CACHE_300}"
  check_file "${TRAIN_CACHE_850}"
  check_file "${TEST_CACHE_850}"
}

run_logged() {
  local name="$1"
  local logfile="$2"
  shift 2
  echo "[AblationRunner] ${name}"
  printf '[AblationRunner] command:'
  printf ' %q' "$@"
  printf '\n'
  mkdir -p "$(dirname "${logfile}")"
  if ! "$@" > "${logfile}" 2>&1; then
    echo "[AblationRunner] FAILED: ${name}; see ${logfile}" | tee -a "${FAILURE_LOG}"
    return 1
  fi
}

write_command() {
  local output_dir="$1"
  shift
  mkdir -p "${output_dir}"
  printf '%q ' "$@" > "${output_dir}/command.txt"
  printf '\n' >> "${output_dir}/command.txt"
}

write_config() {
  local output_dir="$1"
  local split="$2"
  local mode="$3"
  local seed="$4"
  local train_jsonl="$5"
  local test_jsonl="$6"
  local train_cache="$7"
  local test_cache="$8"
  cat > "${output_dir}/config.json" <<EOF
{
  "split": "${split}",
  "ablation_mode": "${mode}",
  "seed": "${seed}",
  "train_jsonl": "${train_jsonl}",
  "test_jsonl": "${test_jsonl}",
  "train_cache": "${train_cache}",
  "test_cache": "${test_cache}",
  "device": "${DEVICE}",
  "epochs_300": "${EPOCHS_300}",
  "epochs_850": "${EPOCHS_850}",
  "batch_size": "${BATCH_SIZE}",
  "lr": "${LR}"
}
EOF
}

rollout_analysis() {
  local name="$1"
  local output_dir="$2"
  run_logged "rollout ${name}" "${LOG_ROOT}/${name}_rollout.log" python3 scripts/analyze_trajectory_rollout.py --records "${name}:${output_dir}/eval_records.jsonl" --dt 0.5 --output_json "${output_dir}/trajectory_rollout_analysis.json" --output_txt "${output_dir}/trajectory_rollout_analysis.txt" --output_samples_jsonl "${output_dir}/trajectory_rollout_samples.jsonl"
}

check_eval() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[AblationRunner] expected eval output missing: ${path}" | tee -a "${FAILURE_LOG}"
    return 1
  fi
}

run_ego_only() {
  local split="$1"
  local train_jsonl="$2"
  local test_jsonl="$3"
  local epochs="$4"
  local output_dir="${OUTPUT_ROOT}/${split}_ego_only_seed${SEED}"
  local train_cmd=(python3 scripts/train_ego_action_head.py --jsonl "${train_jsonl}" --output_dir "${output_dir}" --epochs "${epochs}" --lr "${LR}" --batch_size "${BATCH_SIZE}" --device "${DEVICE}" --max_samples 0)
  local eval_cmd=(python3 scripts/eval_action_heads.py --jsonl "${test_jsonl}" --model_type ego_only --checkpoint "${output_dir}/ego_action_head.pt" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --device "${DEVICE}" --max_samples 0 --ablation_mode ego_only --seed "${SEED}" --output_json "${output_dir}/eval.json" --output_jsonl "${output_dir}/eval_records.jsonl")
  mkdir -p "${output_dir}"
  write_config "${output_dir}" "${split}" "ego_only" "${SEED}" "${train_jsonl}" "${test_jsonl}" "" ""
  write_command "${output_dir}" "${train_cmd[@]}"
  run_logged "${split} ego_only train" "${LOG_ROOT}/${split}_ego_only_seed${SEED}_train.log" "${train_cmd[@]}"
  run_logged "${split} ego_only eval" "${LOG_ROOT}/${split}_ego_only_seed${SEED}_eval.log" "${eval_cmd[@]}"
  check_eval "${output_dir}/eval.json"
  rollout_analysis "${split}_ego_only_seed${SEED}" "${output_dir}"
}

run_vlm_only() {
  local split="$1"
  local train_jsonl="$2"
  local test_jsonl="$3"
  local train_cache="$4"
  local test_cache="$5"
  local epochs="$6"
  local output_dir="${OUTPUT_ROOT}/${split}_vlm_only_seed${SEED}"
  local train_cmd=(python3 scripts/train_action_head.py --jsonl "${train_jsonl}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --output_dir "${output_dir}" --epochs "${epochs}" --lr "${LR}" --batch_size "${BATCH_SIZE}" --device "${DEVICE}" --max_samples 0 --hidden_cache "${train_cache}")
  local eval_cmd=(python3 scripts/eval_action_heads.py --jsonl "${test_jsonl}" --model_type qwen_hidden --checkpoint "${output_dir}/action_head.pt" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --device "${DEVICE}" --hidden_cache "${test_cache}" --max_samples 0 --ablation_mode vlm_only --seed "${SEED}" --output_json "${output_dir}/eval.json" --output_jsonl "${output_dir}/eval_records.jsonl")
  mkdir -p "${output_dir}"
  write_config "${output_dir}" "${split}" "vlm_only" "${SEED}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}"
  write_command "${output_dir}" "${train_cmd[@]}"
  run_logged "${split} vlm_only train" "${LOG_ROOT}/${split}_vlm_only_seed${SEED}_train.log" "${train_cmd[@]}"
  run_logged "${split} vlm_only eval" "${LOG_ROOT}/${split}_vlm_only_seed${SEED}_eval.log" "${eval_cmd[@]}"
  check_eval "${output_dir}/eval.json"
  rollout_analysis "${split}_vlm_only_seed${SEED}" "${output_dir}"
}

run_fusion_ablation() {
  local split="$1"
  local mode="$2"
  local train_jsonl="$3"
  local test_jsonl="$4"
  local train_cache="$5"
  local test_cache="$6"
  local epochs="$7"
  local output_dir="${OUTPUT_ROOT}/${split}_${mode}_seed${SEED}"
  local train_cmd=(python3 scripts/train_fusion_action_head.py --jsonl "${train_jsonl}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --output_dir "${output_dir}" --epochs "${epochs}" --lr "${LR}" --batch_size "${BATCH_SIZE}" --device "${DEVICE}" --max_samples 0 --hidden_cache "${train_cache}" --ablation_mode "${mode}" --seed "${SEED}" --tensorboard_logdir "${TB_ROOT}/${split}_${mode}_seed${SEED}" --step_history_jsonl "${output_dir}/step_history.jsonl" --log_interval_steps 200)
  local eval_cmd=(python3 scripts/eval_action_heads.py --jsonl "${test_jsonl}" --model_type fusion --checkpoint "${output_dir}/fusion_action_head.pt" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --device "${DEVICE}" --hidden_cache "${test_cache}" --ablation_mode "${mode}" --seed "${SEED}" --max_samples 0 --output_json "${output_dir}/eval.json" --output_jsonl "${output_dir}/eval_records.jsonl")
  mkdir -p "${output_dir}"
  write_config "${output_dir}" "${split}" "${mode}" "${SEED}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}"
  write_command "${output_dir}" "${train_cmd[@]}"
  run_logged "${split} ${mode} train" "${LOG_ROOT}/${split}_${mode}_seed${SEED}_train.log" "${train_cmd[@]}"
  run_logged "${split} ${mode} eval" "${LOG_ROOT}/${split}_${mode}_seed${SEED}_eval.log" "${eval_cmd[@]}"
  check_eval "${output_dir}/eval.json"
  rollout_analysis "${split}_${mode}_seed${SEED}" "${output_dir}"
}

run_constant_motion_optional() {
  local split="$1"
  local test_jsonl="$2"
  local mode="$3"
  if [[ ! -f "${test_jsonl}" ]]; then
    echo "[AblationRunner] optional constant_motion ${split}/${mode} skipped: missing ${test_jsonl}" | tee -a "${FAILURE_LOG}"
    return 0
  fi
  local output_dir="${OUTPUT_ROOT}/constant_motion/${split}_${mode}"
  local eval_cmd=(python3 scripts/eval_constant_motion_baseline.py --jsonl "${test_jsonl}" --mode "${mode}" --split "${split}" --output_json "${output_dir}/eval.json" --output_jsonl "${output_dir}/eval_records.jsonl")
  mkdir -p "${output_dir}"
  write_config "${output_dir}" "${split}" "constant_motion_${mode}" "" "" "${test_jsonl}" "" ""
  write_command "${output_dir}" "${eval_cmd[@]}"
  if ! run_logged "constant_motion ${split} ${mode}" "${LOG_ROOT}/constant_motion_${split}_${mode}.log" "${eval_cmd[@]}"; then
    return 0
  fi
  rollout_analysis "constant_motion_${split}_${mode}" "${output_dir}" || true
}

run_tiny_checks() {
  local tiny_root="${OUTPUT_ROOT}/_tiny_checks"
  mkdir -p "${tiny_root}"
  run_logged "tiny ego train" "${LOG_ROOT}/tiny_ego_train.log" python3 scripts/train_ego_action_head.py --jsonl "${TRAIN_JSONL_300}" --output_dir "${tiny_root}/ego" --epochs 1 --lr "${LR}" --batch_size 1 --device "${DEVICE}" --max_samples 2
  run_logged "tiny vlm train" "${LOG_ROOT}/tiny_vlm_train.log" python3 scripts/train_action_head.py --jsonl "${TRAIN_JSONL_300}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --output_dir "${tiny_root}/vlm" --epochs 1 --lr "${LR}" --batch_size 1 --device "${DEVICE}" --max_samples 2 --hidden_cache "${TRAIN_CACHE_300}"
  run_logged "tiny shuffled fusion train" "${LOG_ROOT}/tiny_fusion_train.log" python3 scripts/train_fusion_action_head.py --jsonl "${TRAIN_JSONL_300}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --output_dir "${tiny_root}/fusion_shuffled" --epochs 1 --lr "${LR}" --batch_size 1 --device "${DEVICE}" --max_samples 2 --hidden_cache "${TRAIN_CACHE_300}" --ablation_mode shuffled_vlm --seed "${SEED}"
}

main() {
  echo "[AblationRunner] output_root=${OUTPUT_ROOT}"
  check_required_paths
  if [[ "${RUN_TINY_CHECKS}" == "1" ]]; then
    run_tiny_checks
  fi

  run_ego_only "300" "${TRAIN_JSONL_300}" "${TEST_JSONL_300}" "${EPOCHS_300}"
  run_vlm_only "300" "${TRAIN_JSONL_300}" "${TEST_JSONL_300}" "${TRAIN_CACHE_300}" "${TEST_CACHE_300}" "${EPOCHS_300}"
  run_fusion_ablation "300" "shuffled_vlm" "${TRAIN_JSONL_300}" "${TEST_JSONL_300}" "${TRAIN_CACHE_300}" "${TEST_CACHE_300}" "${EPOCHS_300}"
  run_fusion_ablation "300" "zero_history" "${TRAIN_JSONL_300}" "${TEST_JSONL_300}" "${TRAIN_CACHE_300}" "${TEST_CACHE_300}" "${EPOCHS_300}"
  run_ego_only "850" "${TRAIN_JSONL_850}" "${TEST_JSONL_850}" "${EPOCHS_850}"
  run_vlm_only "850" "${TRAIN_JSONL_850}" "${TEST_JSONL_850}" "${TRAIN_CACHE_850}" "${TEST_CACHE_850}" "${EPOCHS_850}"

  run_constant_motion_optional "100" "${TEST_JSONL_100}" "last"
  run_constant_motion_optional "100" "${TEST_JSONL_100}" "mean"
  run_constant_motion_optional "300" "${TEST_JSONL_300}" "last"
  run_constant_motion_optional "300" "${TEST_JSONL_300}" "mean"
  run_constant_motion_optional "850" "${TEST_JSONL_850}" "last"
  run_constant_motion_optional "850" "${TEST_JSONL_850}" "mean"

  run_logged "summarize ablations" "${LOG_ROOT}/summarize.log" python3 scripts/summarize_paper_ablation_minimal.py --output_root "${OUTPUT_ROOT}"
  echo "[AblationRunner] done. Summary: ${OUTPUT_ROOT}/ablation_summary.md"
}

main "$@"
