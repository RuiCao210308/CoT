#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/root/autodl-tmp/logs_850_logcurve"
mkdir -p "${LOG_DIR}"

echo "[850Logcurve] Fusion logging run"
python3 scripts/train_fusion_action_head.py \
  --jsonl /root/autodl-tmp/action_chunks_gt_waypoint_trainval_850scenes_train.jsonl \
  --model-path qwen \
  --dataroot /root/autodl-tmp/data/nuscenes \
  --output_dir /root/autodl-tmp/fusion_trainval_850scenes_e5_logcurve \
  --epochs 5 \
  --lr 1e-4 \
  --batch_size 1 \
  --device cuda \
  --max_samples 0 \
  --hidden_cache /root/autodl-tmp/qwen_hidden_cache_trainval_850scenes/train_cache.pt \
  --log_interval_steps 200 \
  --step_history_jsonl /root/autodl-tmp/fusion_trainval_850scenes_e5_logcurve/step_history.jsonl \
  2>&1 | tee "${LOG_DIR}/fusion_logcurve.log"

echo "[850Logcurve] Gated-GRAFT seed0 logging run"
python3 scripts/train_frozen_fusion_curvature_residual_head.py \
  --model_type gated_geometry_residual_fusion \
  --jsonl /root/autodl-tmp/action_chunks_gt_waypoint_trainval_850scenes_train.jsonl \
  --model-path qwen \
  --dataroot /root/autodl-tmp/data/nuscenes \
  --fusion_checkpoint /root/autodl-tmp/fusion_trainval_850scenes_e5_real/fusion_action_head.pt \
  --output_dir /root/autodl-tmp/gated_graft_trainval_850scenes_seed0_logcurve \
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
  --step_history_jsonl /root/autodl-tmp/gated_graft_trainval_850scenes_seed0_logcurve/step_history.jsonl \
  2>&1 | tee "${LOG_DIR}/gated_seed0_logcurve.log"

echo "[850Logcurve] Gated-GRAFT seed1 logging run"
python3 scripts/train_frozen_fusion_curvature_residual_head.py \
  --model_type gated_geometry_residual_fusion \
  --jsonl /root/autodl-tmp/action_chunks_gt_waypoint_trainval_850scenes_train.jsonl \
  --model-path qwen \
  --dataroot /root/autodl-tmp/data/nuscenes \
  --fusion_checkpoint /root/autodl-tmp/fusion_trainval_850scenes_e5_real/fusion_action_head.pt \
  --output_dir /root/autodl-tmp/gated_graft_trainval_850scenes_seed1_logcurve \
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
  --step_history_jsonl /root/autodl-tmp/gated_graft_trainval_850scenes_seed1_logcurve/step_history.jsonl \
  2>&1 | tee "${LOG_DIR}/gated_seed1_logcurve.log"

echo "[850Logcurve] Gated-GRAFT seed2 logging run"
python3 scripts/train_frozen_fusion_curvature_residual_head.py \
  --model_type gated_geometry_residual_fusion \
  --jsonl /root/autodl-tmp/action_chunks_gt_waypoint_trainval_850scenes_train.jsonl \
  --model-path qwen \
  --dataroot /root/autodl-tmp/data/nuscenes \
  --fusion_checkpoint /root/autodl-tmp/fusion_trainval_850scenes_e5_real/fusion_action_head.pt \
  --output_dir /root/autodl-tmp/gated_graft_trainval_850scenes_seed2_logcurve \
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
  --step_history_jsonl /root/autodl-tmp/gated_graft_trainval_850scenes_seed2_logcurve/step_history.jsonl \
  2>&1 | tee "${LOG_DIR}/gated_seed2_logcurve.log"

echo "[850Logcurve] Plot training dynamics"
python3 scripts/plot_850_training_dynamics.py \
  --fusion_history /root/autodl-tmp/fusion_trainval_850scenes_e5_logcurve/step_history.jsonl \
  --gated_history /root/autodl-tmp/gated_graft_trainval_850scenes_seed0_logcurve/step_history.jsonl \
  --gated_history /root/autodl-tmp/gated_graft_trainval_850scenes_seed1_logcurve/step_history.jsonl \
  --gated_history /root/autodl-tmp/gated_graft_trainval_850scenes_seed2_logcurve/step_history.jsonl \
  --output_dir /root/autodl-tmp/paper_training_curves_850_final \
  2>&1 | tee "${LOG_DIR}/plot_training_dynamics.log"

echo "[850Logcurve] Done. Figure directory: /root/autodl-tmp/paper_training_curves_850_final"
