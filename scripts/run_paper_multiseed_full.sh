#!/usr/bin/env bash
set -euo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
DATA_ROOT="${DATA_ROOT:-${AUTODL_ROOT}/data/nuscenes}"
CACHE_ROOT="${CACHE_ROOT:-${AUTODL_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AUTODL_ROOT}/openemma_oft_multiseed_full}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
TB_ROOT="${TB_ROOT:-${OUTPUT_ROOT}/tensorboard}"
MODEL_PATH="${MODEL_PATH:-qwen}"
DEVICE="${DEVICE:-cuda}"
MAIN_SEEDS="${MAIN_SEEDS:-0 1 2 3 4}"
ABLATION_SEEDS="${ABLATION_SEEDS:-0 1 2}"
RUN_MAIN_300="${RUN_MAIN_300:-1}"
RUN_MAIN_850="${RUN_MAIN_850:-1}"
RUN_ABLATION_300="${RUN_ABLATION_300:-1}"
RUN_ABLATION_850="${RUN_ABLATION_850:-1}"
RUN_ZERO_HISTORY_850="${RUN_ZERO_HISTORY_850:-1}"
RUN_DETERMINISTIC="${RUN_DETERMINISTIC:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
EPOCHS_300="${EPOCHS_300:-5}"
EPOCHS_850="${EPOCHS_850:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-200}"

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
FAILED_TASKS="${OUTPUT_ROOT}/failed_tasks.txt"
: > "${FAILED_TASKS}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

log_fail() {
  local name="$1"
  local reason="$2"
  echo "${name}: ${reason}" | tee -a "${FAILED_TASKS}"
}

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log_fail "path_check" "missing ${path}"
    exit 1
  fi
}

check_required_paths() {
  if [[ "${BATCH_SIZE}" != "1" ]]; then
    log_fail "config_check" "BATCH_SIZE=${BATCH_SIZE}; train_fusion_action_head.py currently supports BATCH_SIZE=1"
    exit 1
  fi
  if [[ "${RUN_MAIN_300}" == "1" || "${RUN_ABLATION_300}" == "1" ]]; then
    check_file "${TRAIN_JSONL_300}"
    check_file "${TEST_JSONL_300}"
    check_file "${TRAIN_CACHE_300}"
    check_file "${TEST_CACHE_300}"
  fi
  if [[ "${RUN_MAIN_850}" == "1" || "${RUN_ABLATION_850}" == "1" ]]; then
    check_file "${TRAIN_JSONL_850}"
    check_file "${TEST_JSONL_850}"
    check_file "${TRAIN_CACHE_850}"
    check_file "${TEST_CACHE_850}"
  fi
}

append_command() {
  local output_dir="$1"
  local label="$2"
  shift 2
  mkdir -p "${output_dir}"
  {
    echo "## ${label}"
    printf '%q ' "$@"
    printf '\n\n'
  } >> "${output_dir}/command.txt"
}

write_config() {
  local output_dir="$1"
  local family="$2"
  local split="$3"
  local method="$4"
  local seed="$5"
  local train_jsonl="$6"
  local test_jsonl="$7"
  local train_cache="$8"
  local test_cache="$9"
  mkdir -p "${output_dir}"
  cat > "${output_dir}/config.json" <<EOF
{
  "family": "${family}",
  "split": "${split}",
  "method": "${method}",
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

run_cmd() {
  local name="$1"
  local output_dir="$2"
  shift 2
  echo "[MultiseedRunner] ${name}"
  printf '[MultiseedRunner] command:'
  printf ' %q' "$@"
  printf '\n'
  append_command "${output_dir}" "${name}" "$@"
  if ! "$@" >> "${output_dir}/run.log" 2>&1; then
    log_fail "${name}" "command failed; see ${output_dir}/run.log"
    return 1
  fi
}

rollout_analysis() {
  local name="$1"
  local output_dir="$2"
  if [[ ! -f "${output_dir}/eval_records.jsonl" ]]; then
    log_fail "${name}_rollout" "missing eval_records.jsonl"
    return 1
  fi
  if [[ -f "${output_dir}/trajectory_rollout_analysis.json" && "${FORCE_RERUN}" == "0" ]]; then
    echo "[MultiseedRunner] skip rollout existing: ${output_dir}/trajectory_rollout_analysis.json"
    return 0
  fi
  run_cmd "${name} rollout" "${output_dir}" python3 scripts/analyze_trajectory_rollout.py --records "${name}:${output_dir}/eval_records.jsonl" --dt 0.5 --output_json "${output_dir}/trajectory_rollout_analysis.json" --output_txt "${output_dir}/trajectory_rollout_analysis.txt" --output_samples_jsonl "${output_dir}/trajectory_rollout_samples.jsonl"
}

skip_if_eval_exists() {
  local output_dir="$1"
  if [[ -f "${output_dir}/eval.json" && "${FORCE_RERUN}" == "0" ]]; then
    echo "[MultiseedRunner] skip existing eval: ${output_dir}/eval.json"
    return 0
  fi
  return 1
}

existing_override_path() {
  local prefix="$1"
  local split="$2"
  local seed="$3"
  local var_name="${prefix}_${split}_SEED${seed}"
  local value="${!var_name:-}"
  if [[ -n "${value}" && -f "${value}" ]]; then
    echo "${value}"
  fi
}

train_or_eval_fusion() {
  local family="$1"
  local split="$2"
  local seed="$3"
  local train_jsonl="$4"
  local test_jsonl="$5"
  local train_cache="$6"
  local test_cache="$7"
  local epochs="$8"
  local mode="${9:-full}"
  local method="fusion"
  local output_dir="${OUTPUT_ROOT}/${family}/${split}_${method}_seed${seed}"
  if [[ "${mode}" != "full" ]]; then
    output_dir="${OUTPUT_ROOT}/${family}/${split}_${mode}_seed${seed}"
  fi
  mkdir -p "${output_dir}"
  write_config "${output_dir}" "${family}" "${split}" "${mode}" "${seed}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}"
  if skip_if_eval_exists "${output_dir}"; then
    rollout_analysis "${split}_${mode}_seed${seed}" "${output_dir}" || true
    return 0
  fi

  local ckpt="${output_dir}/fusion_action_head.pt"
  local override=""
  if [[ "${mode}" == "full" ]]; then
    override="$(existing_override_path EXISTING_FUSION_CKPT "${split}" "${seed}" || true)"
  fi
  if [[ -n "${override}" ]]; then
    ckpt="${override}"
    echo "[MultiseedRunner] using existing Fusion checkpoint: ${ckpt}"
  fi
  if [[ ! -f "${ckpt}" ]]; then
    run_cmd "${split} ${mode} seed${seed} train" "${output_dir}" python3 scripts/train_fusion_action_head.py --jsonl "${train_jsonl}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --output_dir "${output_dir}" --epochs "${epochs}" --lr "${LR}" --batch_size "${BATCH_SIZE}" --device "${DEVICE}" --max_samples 0 --hidden_cache "${train_cache}" --ablation_mode "${mode}" --seed "${seed}" --tensorboard_logdir "${TB_ROOT}/${family}_${split}_${mode}_seed${seed}" --step_history_jsonl "${output_dir}/step_history.jsonl" --log_interval_steps "${LOG_INTERVAL_STEPS}" || return 1
  else
    echo "[MultiseedRunner] checkpoint exists; eval only: ${ckpt}"
  fi
  run_cmd "${split} ${mode} seed${seed} eval" "${output_dir}" python3 scripts/eval_action_heads.py --jsonl "${test_jsonl}" --model_type fusion --checkpoint "${ckpt}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --device "${DEVICE}" --hidden_cache "${test_cache}" --ablation_mode "${mode}" --seed "${seed}" --max_samples 0 --output_json "${output_dir}/eval.json" --output_jsonl "${output_dir}/eval_records.jsonl" || return 1
  rollout_analysis "${split}_${mode}_seed${seed}" "${output_dir}" || true
}

fusion_checkpoint_for_gated() {
  local split="$1"
  local seed="$2"
  local override
  override="$(existing_override_path EXISTING_FUSION_CKPT "${split}" "${seed}" || true)"
  if [[ -n "${override}" ]]; then
    echo "${override}"
    return 0
  fi
  echo "${OUTPUT_ROOT}/main/${split}_fusion_seed${seed}/fusion_action_head.pt"
}

train_or_eval_gated() {
  local split="$1"
  local seed="$2"
  local train_jsonl="$3"
  local test_jsonl="$4"
  local train_cache="$5"
  local test_cache="$6"
  local epochs="$7"
  local output_dir="${OUTPUT_ROOT}/main/${split}_gated_graft_seed${seed}"
  mkdir -p "${output_dir}"
  write_config "${output_dir}" "main" "${split}" "gated_graft" "${seed}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}"
  if skip_if_eval_exists "${output_dir}"; then
    rollout_analysis "${split}_gated_graft_seed${seed}" "${output_dir}" || true
    return 0
  fi
  local fusion_ckpt
  fusion_ckpt="$(fusion_checkpoint_for_gated "${split}" "${seed}")"
  if [[ ! -f "${fusion_ckpt}" ]]; then
    log_fail "${split}_gated_graft_seed${seed}" "missing Fusion checkpoint for Gated-GRAFT: ${fusion_ckpt}"
    return 1
  fi
  local ckpt="${output_dir}/gated_geometry_residual_fusion_head.pt"
  local override
  override="$(existing_override_path EXISTING_GATED_CKPT "${split}" "${seed}" || true)"
  if [[ -n "${override}" ]]; then
    ckpt="${override}"
    echo "[MultiseedRunner] using existing Gated-GRAFT checkpoint: ${ckpt}"
  fi
  if [[ ! -f "${ckpt}" ]]; then
    run_cmd "${split} gated_graft seed${seed} train" "${output_dir}" python3 scripts/train_frozen_fusion_curvature_residual_head.py --model_type gated_geometry_residual_fusion --jsonl "${train_jsonl}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --fusion_checkpoint "${fusion_ckpt}" --output_dir "${output_dir}" --epochs "${epochs}" --lr "${LR}" --batch_size "${BATCH_SIZE}" --device "${DEVICE}" --hidden_cache "${train_cache}" --geometry_descriptor_weight 0.5 --gate_reg_weight 0.0 --residual_scale 0.1 --seed "${seed}" --tensorboard_logdir "${TB_ROOT}/main_${split}_gated_graft_seed${seed}" --step_history_jsonl "${output_dir}/step_history.jsonl" --log_interval_steps "${LOG_INTERVAL_STEPS}" || return 1
  else
    echo "[MultiseedRunner] checkpoint exists; eval only: ${ckpt}"
  fi
  run_cmd "${split} gated_graft seed${seed} eval" "${output_dir}" python3 scripts/eval_action_heads.py --jsonl "${test_jsonl}" --model_type gated_geometry_residual_fusion --checkpoint "${ckpt}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --device "${DEVICE}" --hidden_cache "${test_cache}" --seed "${seed}" --max_samples 0 --output_json "${output_dir}/eval.json" --output_jsonl "${output_dir}/eval_records.jsonl" || return 1
  rollout_analysis "${split}_gated_graft_seed${seed}" "${output_dir}" || true
}

train_or_eval_ego() {
  local split="$1"
  local seed="$2"
  local train_jsonl="$3"
  local test_jsonl="$4"
  local epochs="$5"
  local output_dir="${OUTPUT_ROOT}/ablation/${split}_ego_only_seed${seed}"
  mkdir -p "${output_dir}"
  write_config "${output_dir}" "ablation" "${split}" "ego_only" "${seed}" "${train_jsonl}" "${test_jsonl}" "" ""
  if skip_if_eval_exists "${output_dir}"; then
    rollout_analysis "${split}_ego_only_seed${seed}" "${output_dir}" || true
    return 0
  fi
  local ckpt="${output_dir}/ego_action_head.pt"
  if [[ ! -f "${ckpt}" ]]; then
    run_cmd "${split} ego_only seed${seed} train" "${output_dir}" python3 scripts/train_ego_action_head.py --jsonl "${train_jsonl}" --output_dir "${output_dir}" --epochs "${epochs}" --lr "${LR}" --batch_size "${BATCH_SIZE}" --device "${DEVICE}" --max_samples 0 --seed "${seed}" || return 1
  else
    echo "[MultiseedRunner] checkpoint exists; eval only: ${ckpt}"
  fi
  run_cmd "${split} ego_only seed${seed} eval" "${output_dir}" python3 scripts/eval_action_heads.py --jsonl "${test_jsonl}" --model_type ego_only --checkpoint "${ckpt}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --device "${DEVICE}" --max_samples 0 --ablation_mode ego_only --seed "${seed}" --output_json "${output_dir}/eval.json" --output_jsonl "${output_dir}/eval_records.jsonl" || return 1
  rollout_analysis "${split}_ego_only_seed${seed}" "${output_dir}" || true
}

train_or_eval_vlm() {
  local split="$1"
  local seed="$2"
  local train_jsonl="$3"
  local test_jsonl="$4"
  local train_cache="$5"
  local test_cache="$6"
  local epochs="$7"
  local output_dir="${OUTPUT_ROOT}/ablation/${split}_vlm_only_seed${seed}"
  mkdir -p "${output_dir}"
  write_config "${output_dir}" "ablation" "${split}" "vlm_only" "${seed}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}"
  if skip_if_eval_exists "${output_dir}"; then
    rollout_analysis "${split}_vlm_only_seed${seed}" "${output_dir}" || true
    return 0
  fi
  local ckpt="${output_dir}/action_head.pt"
  if [[ ! -f "${ckpt}" ]]; then
    run_cmd "${split} vlm_only seed${seed} train" "${output_dir}" python3 scripts/train_action_head.py --jsonl "${train_jsonl}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --output_dir "${output_dir}" --epochs "${epochs}" --lr "${LR}" --batch_size "${BATCH_SIZE}" --device "${DEVICE}" --max_samples 0 --hidden_cache "${train_cache}" --seed "${seed}" || return 1
  else
    echo "[MultiseedRunner] checkpoint exists; eval only: ${ckpt}"
  fi
  run_cmd "${split} vlm_only seed${seed} eval" "${output_dir}" python3 scripts/eval_action_heads.py --jsonl "${test_jsonl}" --model_type qwen_hidden --checkpoint "${ckpt}" --model-path "${MODEL_PATH}" --dataroot "${DATA_ROOT}" --device "${DEVICE}" --hidden_cache "${test_cache}" --max_samples 0 --ablation_mode vlm_only --seed "${seed}" --output_json "${output_dir}/eval.json" --output_jsonl "${output_dir}/eval_records.jsonl" || return 1
  rollout_analysis "${split}_vlm_only_seed${seed}" "${output_dir}" || true
}

run_deterministic() {
  local split="$1"
  local test_jsonl="$2"
  local mode="$3"
  if [[ ! -f "${test_jsonl}" ]]; then
    log_fail "constant_${mode}_${split}" "missing ${test_jsonl}"
    return 1
  fi
  local output_dir="${OUTPUT_ROOT}/deterministic/constant_${mode}_${split}"
  mkdir -p "${output_dir}"
  write_config "${output_dir}" "deterministic" "${split}" "constant_${mode}" "" "" "${test_jsonl}" "" ""
  if skip_if_eval_exists "${output_dir}"; then
    rollout_analysis "constant_${mode}_${split}" "${output_dir}" || true
    return 0
  fi
  run_cmd "constant_${mode}_${split} eval" "${output_dir}" python3 scripts/eval_constant_motion_baseline.py --jsonl "${test_jsonl}" --mode "${mode}" --split "${split}" --output_json "${output_dir}/eval.json" --output_jsonl "${output_dir}/eval_records.jsonl" || return 1
  rollout_analysis "constant_${mode}_${split}" "${output_dir}" || true
}

run_main_split() {
  local split="$1"
  local train_jsonl="$2"
  local test_jsonl="$3"
  local train_cache="$4"
  local test_cache="$5"
  local epochs="$6"
  local seed
  for seed in ${MAIN_SEEDS}; do
    train_or_eval_fusion "main" "${split}" "${seed}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}" "${epochs}" "full" || true
    train_or_eval_gated "${split}" "${seed}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}" "${epochs}" || true
  done
}

run_ablation_split() {
  local split="$1"
  local train_jsonl="$2"
  local test_jsonl="$3"
  local train_cache="$4"
  local test_cache="$5"
  local epochs="$6"
  local run_zero_history="$7"
  local seed
  for seed in ${ABLATION_SEEDS}; do
    train_or_eval_ego "${split}" "${seed}" "${train_jsonl}" "${test_jsonl}" "${epochs}" || true
    train_or_eval_vlm "${split}" "${seed}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}" "${epochs}" || true
    train_or_eval_fusion "ablation" "${split}" "${seed}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}" "${epochs}" "shuffled_vlm" || true
    if [[ "${run_zero_history}" == "1" ]]; then
      train_or_eval_fusion "ablation" "${split}" "${seed}" "${train_jsonl}" "${test_jsonl}" "${train_cache}" "${test_cache}" "${epochs}" "zero_history" || true
    fi
  done
}

main() {
  echo "[MultiseedRunner] output_root=${OUTPUT_ROOT}"
  check_required_paths
  if [[ "${RUN_MAIN_300}" == "1" ]]; then
    run_main_split "300" "${TRAIN_JSONL_300}" "${TEST_JSONL_300}" "${TRAIN_CACHE_300}" "${TEST_CACHE_300}" "${EPOCHS_300}"
  fi
  if [[ "${RUN_MAIN_850}" == "1" ]]; then
    run_main_split "850" "${TRAIN_JSONL_850}" "${TEST_JSONL_850}" "${TRAIN_CACHE_850}" "${TEST_CACHE_850}" "${EPOCHS_850}"
  fi
  if [[ "${RUN_ABLATION_300}" == "1" ]]; then
    run_ablation_split "300" "${TRAIN_JSONL_300}" "${TEST_JSONL_300}" "${TRAIN_CACHE_300}" "${TEST_CACHE_300}" "${EPOCHS_300}" "1"
  fi
  if [[ "${RUN_ABLATION_850}" == "1" ]]; then
    run_ablation_split "850" "${TRAIN_JSONL_850}" "${TEST_JSONL_850}" "${TRAIN_CACHE_850}" "${TEST_CACHE_850}" "${EPOCHS_850}" "${RUN_ZERO_HISTORY_850}"
  fi
  if [[ "${RUN_DETERMINISTIC}" == "1" ]]; then
    run_deterministic "100" "${TEST_JSONL_100}" "last" || true
    run_deterministic "300" "${TEST_JSONL_300}" "last" || true
    run_deterministic "850" "${TEST_JSONL_850}" "last" || true
    run_deterministic "100" "${TEST_JSONL_100}" "mean" || true
    run_deterministic "300" "${TEST_JSONL_300}" "mean" || true
    run_deterministic "850" "${TEST_JSONL_850}" "mean" || true
  fi
  run_cmd "summarize multiseed" "${OUTPUT_ROOT}" python3 scripts/summarize_paper_multiseed_full.py --output_root "${OUTPUT_ROOT}" || true
  echo "[MultiseedRunner] done. Failed tasks: ${FAILED_TASKS}"
}

main "$@"
