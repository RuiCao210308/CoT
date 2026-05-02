#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版 run_tot.py

支持三种算法：
  --method cot   : 原始单次 CoT 预测
  --method sc    : Self-Consistency，多次采样后对数值取平均
  --method tot   : Tree-of-Thought，多意图分支 + 物理启发式打分选优

依赖：
  - NuScenes
  - utils.py: EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, WriteImageSequenceToVideo
  - main.py 中相同的 GenerateMotion 相关依赖（Qwen / LLaVA / GPT）

注意：
  本脚本不再调用 YOLO3D 和 OverlayTrajectory，只做数值轨迹预测 + 可选 2D 轨迹图。
"""
from tqdm import tqdm
import os
import re
import json
import argparse
from datetime import datetime
from math import atan2
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from nuscenes import NuScenes
from pyquaternion import Quaternion

from utils import (
    EstimateCurvatureFromTrajectory,
    IntegrateCurvatureForPoints,
    WriteImageSequenceToVideo,
)
from openemma.planner import (
    PlannerInput,
    build_speed_curvature_prompt,
    build_speed_curvature_retry_prompt,
    build_speed_curvature_sys_message,
    build_qwen_inputs,
    debug_qwen_hidden_shapes,
    evaluate_speed_curvature_prediction,
    extract_qwen_hidden_states,
    format_obs_speed_curvature as planner_format_obs_speed_curvature,
    generate_with_qwen,
    parse_speed_curvature_text as planner_parse_speed_curvature_text,
    select_planning_hidden,
    standardize_speed_curvature_output,
)
from openemma.planner.action_chunk_dataset import (
    append_action_chunk_jsonl,
    make_action_chunk_record,
)

from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from openai import OpenAI
client = OpenAI(api_key="[your-openai-api-key]")

# ----------------- 超参数 -----------------
OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN

# Tree-of-Thought 中使用的候选“驾驶意图”分支
TOT_CANDIDATE_INTENTS = [
    "maintain current speed and follow the lane",
    "slightly slow down and prepare to straighten the car",
    "slightly slow down and follow the current turning direction",
    "maintain speed but reduce curvature to gradually go straight",
    "keep safe distance and be conservative, prefer slowing down slightly",
]


# ------------------------------------------------
# 基础工具函数
# ------------------------------------------------
def parse_speed_curvature_text(raw_text, max_len=FUT_LEN):
    """
    从模型输出中解析 [v, k] 序列。
    raw_text 例如："Future speeds and curvatures: [4.3,-0.5], [4.2,-0.6], ..."
    返回 np.ndarray shape (T, 2)，如果解析失败返回 None。
    """
    return planner_parse_speed_curvature_text(raw_text, max_len=max_len)

def safe_print(msg, use_tqdm):
    if use_tqdm:
        from tqdm import tqdm
        tqdm.write(msg)
    else:
        print(msg)

def pretty_print_step(
    method,
    obs_vel,
    obs_curv,
    pred_speed,
    pred_curv,
    frame_idx=None,
    scene_name=None,
):
    """
    统一格式化并打印：
      - Observed Speed & Curvature
      - Predicted Speed & Curvature
      - First-step Δv / Δk
    """

    # ---------- Observed ----------
    obs_pairs_str = ", ".join(
        [f"[{np.linalg.norm(v):.1f},{c*100:.1f}]" for v, c in zip(obs_vel, obs_curv)]
    )

    # ---------- Predicted ----------
    pred_curv_clean = np.where(np.abs(pred_curv) < 1e-6, 0.0, pred_curv)
    pred_pairs_str = ", ".join(
        [f"[{v:.1f},{k*100:.1f}]" for v, k in zip(pred_speed, pred_curv_clean)]
    )

    # ---------- First-step diff ----------
    last_obs_v = np.linalg.norm(obs_vel[-1])
    last_obs_k = obs_curv[-1] * 100

    dv = abs(pred_speed[0] - last_obs_v)
    dk = abs(pred_curv_clean[0] * 100 - last_obs_k)

    # ---------- Header ----------
    header = f"[{method.upper()}]"
    if scene_name is not None and frame_idx is not None:
        header += f" Scene={scene_name} Frame={frame_idx}"

    #print("=" * 60)
    #print(header)
    print("Observed Speed and Curvature:")
    print(obs_pairs_str)
    print("Predicted Speed and Curvature:")
    print(pred_pairs_str)
    print(f"First-step Δv={dv:.3f}, Δk={dk:.3f}")
    #print("=" * 60)

def average_predictions(pred_list):
    """
    将多次 [T,2] 预测做数值 Self-Consistency 平均。
    pred_list: List[np.ndarray(T,2)]
    返回 np.ndarray(T,2) 或 None
    """
    if len(pred_list) == 0:
        return None
    min_len = min(p.shape[0] for p in pred_list)
    if min_len == 0:
        return None
    stacked = np.stack([p[:min_len] for p in pred_list], axis=0)  # [N,T,2]
    return stacked.mean(axis=0)


def physical_score_candidate(speed_curv, last_speed, last_curv):
    """
    简单物理启发式打分，用于 ToT：
      - 平滑性：相邻 step 的速度/曲率变化不要太大
      - 一致性：首步不要和观测差太多
    返回一个标量得分，越高越好。
    """
    if speed_curv is None or len(speed_curv) == 0:
        return -1e9

    v = speed_curv[:, 0]
    k = speed_curv[:, 1]

    # 与最后观测值的一致性
    diff_v0 = abs(v[0] - last_speed)
    diff_k0 = abs(k[0] - last_curv)

    # 平滑性：相邻差分
    dv = np.abs(np.diff(v))
    dk = np.abs(np.diff(k))

    smooth_v = np.exp(-dv.mean() / 2.0)   # 越小越好
    smooth_k = np.exp(-dk.mean() / 0.5)

    # 合理速度范围惩罚（过大过小惩罚）
    v_penalty = 0.0
    if v.mean() < 0.0:
        v_penalty -= 2.0
    if v.max() > 15.0:
        v_penalty -= 2.0

    # 初始偏差惩罚
    consis = np.exp(-(diff_v0 / 3.0 + diff_k0 / 1.0))

    score = smooth_v + smooth_k + consis + v_penalty
    return float(score)


# ------------------------------------------------
# VLM / CoT 相关（从 main.py 抽出来，做了适当增强）
# ------------------------------------------------
def getMessage(prompt, image=None, args=None):
    if "llama" in args.model_path or "Llama" in args.model_path:
        message = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
    elif "qwen" in args.model_path or "Qwen" in args.model_path:
        message = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
    else:
        # 其他模型简单兜底
        message = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return message


def vlm_inference(text=None, images=None, sys_message=None,
                  processor=None, model=None, tokenizer=None, args=None,
                  temperature=1.0, top_p=0.9, max_new_tokens=256):
    """
    视觉大模型推理统一接口。
    做了两点增强：
      - HF 模型开启 do_sample / temperature / top_p
      - GPT 模型传入 temperature
    """
    # Qwen 系列
    if "qwen" in args.model_path or "Qwen" in args.model_path:
        inputs = build_qwen_inputs(
            prompt=text,
            images=images,
            processor=processor,
            model=model,
            args=args,
            get_message_fn=getMessage,
            process_vision_info_fn=process_vision_info,
        )
        generated_text = generate_with_qwen(
            inputs=inputs,
            processor=processor,
            model=model,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        should_debug_hidden = (
            getattr(args, "debug_hidden_state", False)
            and not getattr(args, "_debug_hidden_state_printed", False)
            and isinstance(text, str)
            and "Future speeds and curvatures:" in text
        )
        if should_debug_hidden:
            try:
                last_hidden_state = extract_qwen_hidden_states(inputs, model)
                planning_hidden = select_planning_hidden(
                    last_hidden_state,
                    inputs["attention_mask"],
                )
                debug_qwen_hidden_shapes(
                    inputs,
                    last_hidden_state,
                    planning_hidden,
                    generated_text,
                )
                setattr(args, "_debug_hidden_state_printed", True)
            except Exception as exc:
                print(f"[QwenHiddenDebug] warning: hidden extraction failed: {exc}")
                setattr(args, "_debug_hidden_state_printed", True)

        return generated_text

    # GPT-4o 模式（多图）
    if "gpt" in args.model_path:
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    *map(lambda x: {"image": x, "resize": 768}, images),
                    text,
                ],
            },
        ]
        if sys_message is not None:
            sys_message_dict = {
                "role": "system",
                "content": sys_message
            }
            PROMPT_MESSAGES.append(sys_message_dict)

        result = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=PROMPT_MESSAGES,
            max_tokens=400,
            temperature=temperature,
        )
        return result.choices[0].message.content

    # 其他模型（llava 等），这里简单兼容，你如果用得到可以按 main.py 再细化
    if "llava" in args.model_path:
        raise NotImplementedError("LLaVA path not fully wired here, 可参考 main.py 补全。")

    return None


def SceneDescription(obs_images, processor=None, model=None, tokenizer=None, args=None):
    prompt = (
        "You are an autonomous driving labeller. You have access to these front-view "
        "camera images of a car taken at a 0.5 second interval over the past 5 seconds. "
        "Imagine you are driving the car. Describe the driving scene according to traffic "
        "lights, movements of other cars or pedestrians and lane markings."
    )
    result = vlm_inference(
        text=prompt, images=obs_images,
        processor=processor, model=model, tokenizer=tokenizer, args=args
    )
    return result


def DescribeObjects(obs_images, processor=None, model=None, tokenizer=None, args=None):
    prompt = (
        "You are an autonomous driving labeller. You have access to a front-view camera "
        "images of a vehicle taken at a 0.5 second interval over the past 5 seconds. "
        "Imagine you are driving the car. What other road users should you pay attention "
        "to in the driving scene? List two or three of them, specifying its location within "
        "the image of the driving scene and provide a short description of the that road user "
        "on what it is doing, and why it is important to you."
    )
    result = vlm_inference(
        text=prompt, images=obs_images,
        processor=processor, model=model, tokenizer=tokenizer, args=args
    )
    return result


def DescribeOrUpdateIntent(obs_images, prev_intent=None,
                           processor=None, model=None, tokenizer=None, args=None):

    if prev_intent is None:
        prompt = (
            "You are an autonomous driving labeller. You have access to a front-view camera "
            "images of a vehicle taken at a 0.5 second interval over the past 5 seconds. "
            "Imagine you are driving the car. Based on the lane markings and the movement "
            "of other cars and pedestrians, describe the desired intent of the ego car. "
            "Is it going to follow the lane to turn left, turn right, or go straight? "
            "Should it maintain the current speed or slow down or speed up?"
        )
    else:
        prompt = (
            f"You are an autonomous driving labeller. You have access to a front-view camera images "
            f"of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving "
            f"the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings "
            f"and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? "
            f"Explain your current intent."
        )

    result = vlm_inference(
        text=prompt, images=obs_images,
        processor=processor, model=model, tokenizer=tokenizer, args=args
    )
    return result


def build_sys_message():
    return build_speed_curvature_sys_message(article="an")


def format_obs_speed_curvature(obs_velocities, obs_curvatures):
    return planner_format_obs_speed_curvature(obs_velocities, obs_curvatures)


def generate_motion_single(obs_images, obs_velocities, obs_curvatures,
                           processor=None, model=None, tokenizer=None, args=None,
                           extra_intent_text=None,
                           temperature=1.0, top_p=0.9,
                           return_metadata=False):
    """
    单次 CoT 预测，用于：
      - baseline CoT
      - SC 的子采样
      - ToT 的分支样本
    返回：raw_text, scene_desc, object_desc, intent_desc
    """
    scene_description = SceneDescription(obs_images, processor, model, tokenizer, args)
    object_description = DescribeObjects(obs_images, processor, model, tokenizer, args)
    intent_description = DescribeOrUpdateIntent(obs_images, None, processor, model, tokenizer, args)

    sys_message = build_sys_message()

    planner_input = PlannerInput(
        images=obs_images,
        obs_velocities=obs_velocities,
        obs_curvatures=obs_curvatures,
        scene_description=scene_description,
        object_description=object_description,
        intent_description=intent_description,
        method=args.method,
        reasoning_mode=getattr(args, "reasoning_mode", "cot"),
        extra_intent_text=extra_intent_text or "",
        horizon=FUT_LEN,
    )
    prompt, obs_speed_curvature_str = build_speed_curvature_prompt(planner_input)
    #print(f"Observed Speed and Curvature: {obs_speed_curvature_str}")

    raw = None
    guard_details = None
    initial_guard_details = None
    retry_used = False
    retry_reason = None
    retry_count = 0
    final_planning_prompt = prompt
    max_planner_retries = 3
    for _ in range(max_planner_retries):
        raw = vlm_inference(
            text=prompt,
            images=obs_images,
            sys_message=sys_message,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            args=args,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=256,
        )
        arr = parse_speed_curvature_text(raw)
        guard_details = evaluate_speed_curvature_prediction(
            arr,
            obs_velocities,
            obs_curvatures,
            return_details=True,
        )
        if initial_guard_details is None:
            initial_guard_details = dict(guard_details)
        is_valid = guard_details["is_valid"]
        reject_reasons = guard_details["reasons"]
        if isinstance(raw, str) and "[" in raw and is_valid:
            break
        if isinstance(raw, str) and "[" in raw:
            print(f"[PlannerGuard] retrying rejected prediction: {', '.join(reject_reasons)}")
            retry_used = True
            retry_count += 1
            retry_reason = ", ".join(reject_reasons)
            retry_prompt = build_speed_curvature_retry_prompt(prompt, raw, reject_reasons)
            retry_raw = vlm_inference(
                text=retry_prompt,
                images=obs_images,
                sys_message=sys_message,
                processor=processor,
                model=model,
                tokenizer=tokenizer,
                args=args,
                temperature=max(temperature, 0.8),
                top_p=top_p,
                max_new_tokens=256,
            )
            retry_arr = parse_speed_curvature_text(retry_raw)
            retry_guard_details = evaluate_speed_curvature_prediction(
                retry_arr,
                obs_velocities,
                obs_curvatures,
                return_details=True,
            )
            retry_valid = retry_guard_details["is_valid"]
            if isinstance(retry_raw, str) and "[" in retry_raw:
                raw = retry_raw
                guard_details = retry_guard_details
                final_planning_prompt = retry_prompt
            if retry_valid:
                break
    if guard_details is None:
        guard_details = evaluate_speed_curvature_prediction(
            None,
            obs_velocities,
            obs_curvatures,
            return_details=True,
        )
    if initial_guard_details is None:
        initial_guard_details = dict(guard_details)
    guard_details = dict(guard_details)
    guard_details.update(
        {
            "initial_is_valid": bool(initial_guard_details.get("is_valid", False)),
            "initial_reject_reasons": list(initial_guard_details.get("reasons", [])),
            "final_prediction_valid": bool(guard_details.get("is_valid", False)),
            "final_reject_reasons": [] if guard_details.get("is_valid", False) else list(guard_details.get("reasons", [])),
            "retry_used": bool(retry_used),
            "retry_count": int(retry_count),
        }
    )
    metadata = {
        "raw_qwen_text": raw,
        "retry_used": retry_used,
        "retry_reason": retry_reason,
        "planner_guard": guard_details,
        "system_message": sys_message,
        "planning_prompt": final_planning_prompt,
        "prompt_type": "speed_curvature_planning",
    }
    if return_metadata:
        return raw, scene_description, object_description, intent_description, metadata
    return raw, scene_description, object_description, intent_description


# ------------------------------------------------
# SC / ToT 封装
# ------------------------------------------------
def predict_with_cot(obs_images, obs_velocities, obs_curvatures,
                     processor, model, tokenizer, args):
    raw, scene_desc, obj_desc, intent_desc, pred_meta = generate_motion_single(
        obs_images, obs_velocities, obs_curvatures,
        processor=processor, model=model, tokenizer=tokenizer, args=args,
        temperature=0.7, top_p=0.9,
        return_metadata=True,
    )
    arr = parse_speed_curvature_text(raw)
    return arr, scene_desc, obj_desc, intent_desc, pred_meta


def predict_with_sc(obs_images, obs_velocities, obs_curvatures,
                    processor, model, tokenizer, args,
                    sc_samples=5):
    preds = []
    first_scene = first_obj = first_intent = None
    first_meta = None

    for i in range(sc_samples):
        raw, scene_desc, obj_desc, intent_desc, pred_meta = generate_motion_single(
            obs_images, obs_velocities, obs_curvatures,
            processor=processor, model=model, tokenizer=tokenizer, args=args,
            temperature=1.0, top_p=0.9,
            return_metadata=True,
        )
        arr = parse_speed_curvature_text(raw)
        if arr is not None:
            preds.append(arr)
            if first_scene is None:
                first_scene, first_obj, first_intent = scene_desc, obj_desc, intent_desc
                first_meta = pred_meta

    if not preds:
        return None, None, None, None, None

    mean_pred = average_predictions(preds)
    if first_meta is not None:
        first_meta = dict(first_meta)
        first_meta["raw_qwen_text"] = first_meta.get("raw_qwen_text")
        first_meta["sc_samples"] = len(preds)
    return mean_pred, first_scene, first_obj, first_intent, first_meta


def predict_with_tot(obs_images, obs_velocities, obs_curvatures,
                     processor, model, tokenizer, args):
    """
    简化版 ToT：
      - 构造多个“驾驶意图”分支 extra_intent_text
      - 每个分支调用一次 generate_motion_single
      - 用物理启发式打分，选得分最高的那条预测
    """
    obs_speed_curvature_str, last_v, last_k = format_obs_speed_curvature(
        obs_velocities, obs_curvatures
    )

    best_score = -1e9
    best_pred = None
    best_scene = best_obj = best_intent = None
    best_meta = None

    for intent_hint in TOT_CANDIDATE_INTENTS:
        extra_text = f"Assume the driver intent is: {intent_hint}."
        raw, scene_desc, obj_desc, intent_desc, pred_meta = generate_motion_single(
            obs_images, obs_velocities, obs_curvatures,
            processor=processor, model=model, tokenizer=tokenizer, args=args,
            extra_intent_text=extra_text,
            temperature=0.9, top_p=0.9,
            return_metadata=True,
        )
        arr = parse_speed_curvature_text(raw)
        if arr is None:
            continue
        score = physical_score_candidate(arr, last_v, last_k)
        print(f"[ToT] intent='{intent_hint}' score={score:.3f}")
        if score > best_score:
            best_score = score
            best_pred = arr
            best_scene, best_obj, best_intent = scene_desc, obj_desc, intent_desc
            best_meta = dict(pred_meta)
            best_meta["tot_intent_hint"] = intent_hint
            best_meta["tot_score"] = score
        print(best_pred)
    if best_pred is None:
        return None, None, None, None, None
    return best_pred, best_scene, best_obj, best_intent, best_meta


# ------------------------------------------------
# 模型加载（简化版本：主要考虑 Qwen & GPT）
# ------------------------------------------------
def load_model_from_args(args):
    model = None
    processor = None
    tokenizer = None

    if "qwen" in args.model_path or "Qwen" in args.model_path:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        os.environ["HF_HOME"] = "/home/Cr_seu0321/.cache/huggingface"
        os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/Cr_seu0321/.cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = "/home/Cr_seu0321/.cache/huggingface"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        LOCAL_ID = "/home/Cr_seu0321/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
                    LOCAL_ID,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    )           
        processor = AutoProcessor.from_pretrained(
                            LOCAL_ID,
                            local_files_only=True
                    )
        tokenizer = None
        return model, processor, tokenizer

# ------------------------------------------------
# 主评估循环
# ------------------------------------------------
def main_loop(args, model, processor, tokenizer):
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = nusc.scene
    print(f"Number of scenes: {len(scenes)}")

    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    out_root = f"{args.model_path}_results/{args.method}/{timestamp}"
    os.makedirs(out_root, exist_ok=True)
    action_chunk_jsonl_path = args.action_chunk_output
    if args.save_action_chunks and not action_chunk_jsonl_path:
        action_chunk_jsonl_path = os.path.join(out_root, "action_chunks.jsonl")
    if args.save_action_chunks:
        print("Action chunk JSONL:", action_chunk_jsonl_path)
    excel_rows = []#excel容器
    all_cam_images_sequence = []
   
    for scene_idx, scene in enumerate(scenes):
        token = scene["token"]
        first_sample_token = scene["first_sample_token"]
        last_sample_token = scene["last_sample_token"]
        name = scene["name"]
        description = scene["description"]

        # 你可以只跑特定 scene，方便 debug
        if name  in ["scene-0553"]:
            continue

        print("\n🚗 Processing scene:", name)
        front_camera_images = []
        ego_poses = []
        camera_params = []
        sample_tokens = []
        sample_timestamps = []

        curr_sample_token = first_sample_token
        while True:
            sample = nusc.get("sample", curr_sample_token)
            cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            img_path = os.path.join(nusc.dataroot, cam_front_data["filename"])
            front_camera_images.append(img_path)
            sample_tokens.append(curr_sample_token)
            sample_timestamps.append(sample.get("timestamp", cam_front_data.get("timestamp")))

            pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            ego_poses.append(pose)

            camera_params.append(nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"]))

            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample["next"]

        scene_length = len(front_camera_images)
        print(f"   → scene length = {scene_length}")
        if scene_length < TTL_LEN:
            print(f"   → too short (<{TTL_LEN}), skip.")
            continue

        ego_poses_world = np.array([p["translation"][:3] for p in ego_poses])
        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)

        # Debug：重建轨迹
        if args.plot:
            est_points = IntegrateCurvatureForPoints(
                ego_curvatures,
                np.linalg.norm(ego_velocities, axis=1),
                ego_poses_world[0],
                atan2(ego_velocities[0][1], ego_velocities[0][0]),
                scene_length,
            )
            plt.figure()
            plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], "r-", label="GT")
            plt.plot(est_points[:, 0], est_points[:, 1], "g-", label="Reconstruct")
            plt.legend()
            plt.title(f"{name} interpolation")
            plt.savefig(os.path.join(out_root, f"{name}_interpolation.jpg"))
            plt.close()

        ego_traj_world = [p["translation"][:3] for p in ego_poses]

        ade1s_list, ade2s_list, ade3s_list = [], [], []
        cam_images_sequence = []

        for i in range(scene_length - TTL_LEN):
            obs_images_paths = front_camera_images[i : i + OBS_LEN]
            obs_poses = ego_poses[i : i + OBS_LEN]
            obs_cam_params = camera_params[i : i + OBS_LEN]
            obs_traj_world = ego_traj_world[i : i + OBS_LEN]
            fut_traj_world = ego_traj_world[i + OBS_LEN : i + TTL_LEN]
            obs_vel = ego_velocities[i : i + OBS_LEN]
            obs_curv = ego_curvatures[i : i + OBS_LEN]
            fut_vel = ego_velocities[i + OBS_LEN : i + TTL_LEN]
            fut_curv = ego_curvatures[i + OBS_LEN : i + TTL_LEN]

            # 对于 Qwen，传最后一帧图片路径；对于 GPT，要用 base64 列表，这里不区分，简单用最后一帧路径
            curr_image_path = obs_images_paths[-1]
            if "gpt" in args.model_path:
                with open(curr_image_path, "rb") as f:
                    img_b64 = [base64.b64encode(f.read()).decode("utf-8")]
                obs_images_for_vlm = img_b64
            else:
                obs_images_for_vlm = curr_image_path

            # ------------------ 根据 method 选择算法 ------------------
            if args.method == "cot":
                pred_vk, scene_desc, obj_desc, intent_desc, pred_meta = predict_with_cot(
                    obs_images_for_vlm, obs_vel, obs_curv, processor, model, tokenizer, args
                )
            elif args.method == "sc":
                pred_vk, scene_desc, obj_desc, intent_desc, pred_meta = predict_with_sc(
                    obs_images_for_vlm, obs_vel, obs_curv, processor, model, tokenizer, args,
                    sc_samples=args.sc_samples
                )
            elif args.method == "tot":
                pred_vk, scene_desc, obj_desc, intent_desc, pred_meta = predict_with_tot(
                    obs_images_for_vlm, obs_vel, obs_curv, processor, model, tokenizer, args
                )
            else:
                raise ValueError(f"Unknown method: {args.method}")

            if pred_vk is None:
                print(f"   → frame {i}: parse failed, skip.")
                continue

            planner_output = standardize_speed_curvature_output(
                speed_curvature=pred_vk,
                initial_position=fut_traj_world[0],
                initial_heading=atan2(ego_velocities[i + OBS_LEN - 1][1],
                                      ego_velocities[i + OBS_LEN - 1][0]),
                max_len=FUT_LEN,
                scene_description=scene_desc,
                object_description=obj_desc,
                intent_description=intent_desc,
            )
            pred_curv = planner_output.curvatures
            pred_speed = planner_output.speeds
            #print预测
            pred_len = min(FUT_LEN, len(pred_curv))
            pred_pairs_str = ",".join([f"[{v:.2f},{k*100:.2f}]" for v,k in zip(pred_speed[:pred_len], pred_curv[:pred_len])])
            obs_pairs_str = ",".join([f"[{np.linalg.norm(v):.2f},{c*100:.2f}]" for v,c in zip(obs_vel, obs_curv)])
        
            fut_traj_world_np = np.array(fut_traj_world)

            # 轨迹积分
            pred_traj = planner_output.trajectory

            # ADE 指标
            ade_all = np.mean(np.linalg.norm(fut_traj_world_np[:pred_len] - pred_traj[:pred_len], axis=1))

            pred1_len = min(pred_len, 2)
            ade1 = np.mean(
                np.linalg.norm(fut_traj_world_np[:pred1_len] - pred_traj[1 : pred1_len + 1], axis=1)
            )
            ade1s_list.append(ade1)

            pred2_len = min(pred_len, 4)
            ade2 = np.mean(
                np.linalg.norm(fut_traj_world_np[:pred2_len] - pred_traj[:pred2_len], axis=1)
            )
            ade2s_list.append(ade2)

            pred3_len = min(pred_len, 6)
            ade3 = np.mean(
                np.linalg.norm(fut_traj_world_np[:pred3_len] - pred_traj[:pred3_len], axis=1)
            )
            ade3s_list.append(ade3)
            last_obs_v = np.linalg.norm(obs_vel[-1])
            last_obs_k = obs_curv[-1]
            first_step_delta_v = abs(pred_speed[0] - last_obs_v)
            first_step_delta_k = abs(pred_curv[0] - last_obs_k)
            print(f"   → frame {i}: ADE={ade_all:.3f}, ADE1s={ade1:.3f}, ADE2s={ade2:.3f}, ADE3s={ade3:.3f}")
            pretty_print_step(
                method=args.method,
                obs_vel=obs_vel,
                obs_curv=obs_curv,
                pred_speed=pred_speed[:pred_len],
                pred_curv=pred_curv[:pred_len],
                frame_idx=i,
                scene_name=name,
            )

            if args.save_action_chunks:
                try:
                    future_action_gt = np.stack(
                        [np.linalg.norm(fut_vel, axis=1), fut_curv],
                        axis=1,
                    )
                    pred_action = np.stack([pred_speed[:pred_len], pred_curv[:pred_len]], axis=1)
                    record = make_action_chunk_record(
                        scene_name=name,
                        scene_index=scene_idx,
                        frame_idx=i,
                        sample_token=sample_tokens[i + OBS_LEN] if i + OBS_LEN < len(sample_tokens) else None,
                        timestamp=sample_timestamps[i + OBS_LEN] if i + OBS_LEN < len(sample_timestamps) else None,
                        method=args.method,
                        model_path=args.model_path,
                        obs_velocities=obs_vel,
                        obs_curvatures=obs_curv,
                        future_speed_curvature_gt=future_action_gt[:FUT_LEN],
                        predicted_speed_curvature=pred_action[:FUT_LEN],
                        raw_qwen_text=(pred_meta or {}).get("raw_qwen_text"),
                        retry_used=(pred_meta or {}).get("retry_used", False),
                        retry_reason=(pred_meta or {}).get("retry_reason"),
                        system_message=(pred_meta or {}).get("system_message"),
                        planning_prompt=(pred_meta or {}).get("planning_prompt"),
                        prompt_type=(pred_meta or {}).get("prompt_type", "speed_curvature_planning"),
                        planner_guard=(pred_meta or {}).get("planner_guard"),
                        metrics={
                            "ade": ade_all,
                            "ade_1s": ade1,
                            "ade_2s": ade2,
                            "ade_3s": ade3,
                            "first_step_delta_v": first_step_delta_v,
                            "first_step_delta_k": first_step_delta_k,
                        },
                        extra_metadata={
                            "scene_token": token,
                            "image_path": curr_image_path,
                        },
                    )
                    append_action_chunk_jsonl(record, action_chunk_jsonl_path)
                except Exception as e:
                    print(f"[ActionChunkLogger] warning: failed to save frame {name}/{i}: {e}")

            # 可视化保存（不依赖 YOLO3D 和 OverlayTrajectory）
            if args.plot:
                img = cv2.imread(curr_image_path)
                if img is not None:
                    cam_images_sequence.append(img.copy())
                    cv2.imwrite(os.path.join(out_root, f"{name}_{i}_front_cam.jpg"), img)

                plt.figure()
                plt.plot(fut_traj_world_np[:, 0], fut_traj_world_np[:, 1], "r-", label="GT")
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", label="Pred")
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade_all:.3f}")
                plt.savefig(os.path.join(out_root, f"{name}_{i}_traj.jpg"))
                plt.close()

                np.save(os.path.join(out_root, f"{name}_{i}_pred_traj.npy"), pred_traj)
                np.save(os.path.join(out_root, f"{name}_{i}_pred_curv.npy"), pred_curv)
                np.save(os.path.join(out_root, f"{name}_{i}_pred_speed.npy"), pred_speed)

                with open(os.path.join(out_root, f"{name}_{i}_logs.txt"), "w") as f:
                    f.write(f"Scene: {name}\n")
                    f.write(f"Scene Description: {scene_desc}\n")
                    f.write(f"Object Description: {obj_desc}\n")
                    f.write(f"Intent Description: {intent_desc}\n")
                    f.write(f"Observed speed and curvature(v,k*100):\n")
                    f.write(obs_pairs_str+"\n\n")
                    f.write(f"Predicted speed and curvature(v,k*100):\n")
                    f.write(pred_pairs_str+"\n\n")

                    last_obs_v = np.linalg.norm(obs_vel[-1])
                    last_obs_k_x100 = obs_curv[-1] * 100
                    pred_k_x100 = pred_curv[0] * 100
                    f.write("First-step difference:\n")
                    f.write(f"delta v = {abs(pred_speed[0] - last_obs_v):.3f}\n")
                    f.write(f"delta k_x100 = {abs(pred_k_x100 - last_obs_k_x100):.3f}\n\n")

                    f.write(f"ADE_all: {ade_all}\n")
                    f.write(f"ADE1s: {ade1}\n")
                    f.write(f"ADE2s: {ade2}\n")
                    f.write(f"ADE3s: {ade3}\n")

        # 每个 scene 汇总
        if len(ade1s_list) == 0:
            print(f"Scene {name}: no valid predictions, skip metrics.")
            continue

        mean_ade1s = float(np.mean(ade1s_list))
        mean_ade2s = float(np.mean(ade2s_list))
        mean_ade3s = float(np.mean(ade3s_list))
        aveg_ade = float(np.mean([mean_ade1s, mean_ade2s, mean_ade3s]))
        excel_rows.append({
                "alg":args.method,
                "scene": name,
                "ade1s": ade1,
                "ade2s": ade2,
                "ade3s": ade3,
                "avgade": ade_all,
            })
        scene_result = {
            "name": name,
            "token": token,
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": aveg_ade,
        }
        print(f"Scene {name} summary:", scene_result)

        with open(os.path.join(out_root, "ade_results.jsonl"), "a") as f:
            f.write(json.dumps(scene_result) + "\n")

        if args.plot and len(cam_images_sequence) > 0:
            WriteImageSequenceToVideo(cam_images_sequence, os.path.join(out_root, name))
    import pandas as pd
    excel_path = os.path.join(out_root, "ade_summary.xlsx")
    df = pd.DataFrame(excel_rows)
    df.to_excel(excel_path, index=False)
    print("✅ Excel saved to:", excel_path)

    print("\n✅ Done. Results saved to:", out_root)


# ------------------------------------------------
# 入口
# ------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qwen",
                        help="qwen / gpt / (llava: 需要你自己补全)")
    parser.add_argument("--dataroot", type=str, default="/home/Cr_seu0321/data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--plot", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--method", type=str, default="cot",
                        choices=["cot", "sc", "tot"],
                        help="cot: baseline; sc: self-consistency; tot: tree-of-thought")
    parser.add_argument("--sc-samples", type=int, default=5,
                        help="self-consistency 采样次数")
    parser.add_argument("--save_action_chunks", type=lambda x: str(x).lower() == "true", default=False,
                        help="True: append per-frame action chunk training records to JSONL")
    parser.add_argument("--action_chunk_output", type=str, default=None,
                        help="Path to action chunk JSONL. Defaults to <result_dir>/action_chunks.jsonl")
    parser.add_argument("--debug_hidden_state", type=lambda x: str(x).lower() == "true", default=False,
                        help="True: print one Qwen hidden-state shape summary for the planning prompt")
    parser.add_argument(
    "--use-tqdm",
    type=lambda x: str(x).lower() == "true",
    default=True,
    help="True: show progress bar only; False: verbose debug output"
)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Args:", args)

    model, processor, tokenizer = load_model_from_args(args)
    main_loop(args, model, processor, tokenizer)
