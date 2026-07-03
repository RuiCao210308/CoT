#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/root/autodl-tmp/logs_850_tensorboard_logcurve"
mkdir -p "${LOG_DIR}"

run_step() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo "[850TensorBoardLogcurve] ${name}"
  set +e
  "$@" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "[850TensorBoardLogcurve] failed: ${name}"
    echo "[850TensorBoardLogcurve] see log: ${log_file}"
    exit "${status}"
  fi
}

run_step "Fusion-850 logging run" "${LOG_DIR}/fusion_tbcurve.log" \
  python3 scripts/train_fusion_action_head.py \
    --jsonl /root/autodl-tmp/action_chunks_gt_waypoint_trainval_850scenes_train.jsonl \
    --model-path qwen \
    --dataroot /root/autodl-tmp/data/nuscenes \
    --output_dir /root/autodl-tmp/fusion_trainval_850scenes_e5_tbcurve \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 1 \
    --device cuda \
    --max_samples 0 \
    --hidden_cache /root/autodl-tmp/qwen_hidden_cache_trainval_850scenes/train_cache.pt \
    --log_interval_steps 200 \
    --tensorboard_logdir /root/autodl-tmp/tensorboard_runs_850/Fusion-850 \
    --step_history_jsonl /root/autodl-tmp/fusion_trainval_850scenes_e5_tbcurve/step_history.jsonl

run_step "Gated-GRAFT-850 seed0 logging run" "${LOG_DIR}/gated_seed0_tbcurve.log" \
  python3 scripts/train_frozen_fusion_curvature_residual_head.py \
    --model_type gated_geometry_residual_fusion \
    --jsonl /root/autodl-tmp/action_chunks_gt_waypoint_trainval_850scenes_train.jsonl \
    --model-path qwen \
    --dataroot /root/autodl-tmp/data/nuscenes \
    --fusion_checkpoint /root/autodl-tmp/fusion_trainval_850scenes_e5_real/fusion_action_head.pt \
    --output_dir /root/autodl-tmp/gated_graft_trainval_850scenes_seed0_tbcurve \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 1 \
    --device cuda \
    --hidden_cache /root/autodl-tmp/qwen_hidden_cache_trainval_850scenes/train_cache.pt \
    --geometry_descriptor_weight 0.5 \
    --gate_reg_weight 0.0 \
    --residual_scale 0.1 \
    --seed 0 \
    --log_interval_steps 200 \
    --tensorboard_logdir /root/autodl-tmp/tensorboard_runs_850/Gated-GRAFT-850_seed0 \
    --step_history_jsonl /root/autodl-tmp/gated_graft_trainval_850scenes_seed0_tbcurve/step_history.jsonl

run_step "Gated-GRAFT-850 seed1 logging run" "${LOG_DIR}/gated_seed1_tbcurve.log" \
  python3 scripts/train_frozen_fusion_curvature_residual_head.py \
    --model_type gated_geometry_residual_fusion \
    --jsonl /root/autodl-tmp/action_chunks_gt_waypoint_trainval_850scenes_train.jsonl \
    --model-path qwen \
    --dataroot /root/autodl-tmp/data/nuscenes \
    --fusion_checkpoint /root/autodl-tmp/fusion_trainval_850scenes_e5_real/fusion_action_head.pt \
    --output_dir /root/autodl-tmp/gated_graft_trainval_850scenes_seed1_tbcurve \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 1 \
    --device cuda \
    --hidden_cache /root/autodl-tmp/qwen_hidden_cache_trainval_850scenes/train_cache.pt \
    --geometry_descriptor_weight 0.5 \
    --gate_reg_weight 0.0 \
    --residual_scale 0.1 \
    --seed 1 \
    --log_interval_steps 200 \
    --tensorboard_logdir /root/autodl-tmp/tensorboard_runs_850/Gated-GRAFT-850_seed1 \
    --step_history_jsonl /root/autodl-tmp/gated_graft_trainval_850scenes_seed1_tbcurve/step_history.jsonl

run_step "Gated-GRAFT-850 seed2 logging run" "${LOG_DIR}/gated_seed2_tbcurve.log" \
  python3 scripts/train_frozen_fusion_curvature_residual_head.py \
    --model_type gated_geometry_residual_fusion \
    --jsonl /root/autodl-tmp/action_chunks_gt_waypoint_trainval_850scenes_train.jsonl \
    --model-path qwen \
    --dataroot /root/autodl-tmp/data/nuscenes \
    --fusion_checkpoint /root/autodl-tmp/fusion_trainval_850scenes_e5_real/fusion_action_head.pt \
    --output_dir /root/autodl-tmp/gated_graft_trainval_850scenes_seed2_tbcurve \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 1 \
    --device cuda \
    --hidden_cache /root/autodl-tmp/qwen_hidden_cache_trainval_850scenes/train_cache.pt \
    --geometry_descriptor_weight 0.5 \
    --gate_reg_weight 0.0 \
    --residual_scale 0.1 \
    --seed 2 \
    --log_interval_steps 200 \
    --tensorboard_logdir /root/autodl-tmp/tensorboard_runs_850/Gated-GRAFT-850_seed2 \
    --step_history_jsonl /root/autodl-tmp/gated_graft_trainval_850scenes_seed2_tbcurve/step_history.jsonl

run_step "Plot TensorBoard training curves" "${LOG_DIR}/plot_tensorboard_training_curves.log" \
  python3 scripts/plot_tensorboard_training_curves.py \
    --runs Fusion-850:/root/autodl-tmp/tensorboard_runs_850/Fusion-850 \
    --runs Gated-GRAFT-850:/root/autodl-tmp/tensorboard_runs_850/Gated-GRAFT-850_seed0 \
    --runs Gated-GRAFT-850:/root/autodl-tmp/tensorboard_runs_850/Gated-GRAFT-850_seed1 \
    --runs Gated-GRAFT-850:/root/autodl-tmp/tensorboard_runs_850/Gated-GRAFT-850_seed2 \
    --output_dir /root/autodl-tmp/paper_training_curves_850_tensorboard \
    --metrics train/speed_l1 train/curvature_l1 train/gate_mean \
    --smooth_window 5 \
    --alignment interpolate

echo "[850TensorBoardLogcurve] Done."
echo "[850TensorBoardLogcurve] TensorBoard logdir: /root/autodl-tmp/tensorboard_runs_850"
echo "[850TensorBoardLogcurve] Figure directory: /root/autodl-tmp/paper_training_curves_850_tensorboard"
