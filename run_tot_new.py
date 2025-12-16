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
    if not isinstance(raw_text, str):
        return None

    # 干掉前缀
    raw_text = raw_text.replace("Future speeds and curvatures:", "")
    coords = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", raw_text)
    if not coords:
        return None

    pairs = []
    for v, k in coords:
        try:
            v_f = float(v)
            k_f = float(k)
            pairs.append([v_f, k_f])
        except Exception:
            continue

    if not pairs:
        return None

    arr = np.array(pairs, dtype=np.float32)
    if arr.shape[0] > max_len:
        arr = arr[:max_len]
    return arr

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
        message = getMessage(text, image=images, args=args)
        text_prompt = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

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
    sys_message = (
        "You are an autonomous driving labeller. You have access to a front-view camera image of a vehicle, "
        "a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature "
        "is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. "
        "A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. "
        "The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is "
        "driving straight. As a driver on the road, you should follow any common sense traffic rules. "
        "You should try to stay in the middle of your lane. You should maintain necessary distance from "
        "the leading vehicle. You should observe lane markings and follow them.  Your task is to do your "
        "best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given "
        "vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. "
        "If you cannot provide a response people will get injured."
    )
    return sys_message


def format_obs_speed_curvature(obs_velocities, obs_curvatures):
    # 速度模长
    obs_vel_norm = np.linalg.norm(obs_velocities, axis=1)
    # 曲率 *100
    obs_curv_scaled = obs_curvatures * 100.0
    pairs = [f"[{v:.1f},{k:.1f}]" for v, k in zip(obs_vel_norm, obs_curv_scaled)]
    return ", ".join(pairs), obs_vel_norm[-1], obs_curv_scaled[-1]


def generate_motion_single(obs_images, obs_velocities, obs_curvatures,
                           processor=None, model=None, tokenizer=None, args=None,
                           extra_intent_text=None,
                           temperature=1.0, top_p=0.9):
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

    obs_speed_curvature_str, _, _ = format_obs_speed_curvature(obs_velocities, obs_curvatures)
    #print(f"Observed Speed and Curvature: {obs_speed_curvature_str}")

    sys_message = build_sys_message()

    if extra_intent_text is None:
        extra_intent_text = ""

    prompt = f"""
These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
The scene is described as follows: {scene_description}. 
The identified critical objects are {object_description}. 
The car's intent is {intent_description}. {extra_intent_text}
The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format
[speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. 
Write the raw text not markdown or latex. Future speeds and curvatures:
    """.strip()

    raw = None
    for _ in range(3):
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
        if isinstance(raw, str) and "[" in raw:
            break
    return raw, scene_description, object_description, intent_description


# ------------------------------------------------
# SC / ToT 封装
# ------------------------------------------------
def predict_with_cot(obs_images, obs_velocities, obs_curvatures,
                     processor, model, tokenizer, args):
    raw, scene_desc, obj_desc, intent_desc = generate_motion_single(
        obs_images, obs_velocities, obs_curvatures,
        processor=processor, model=model, tokenizer=tokenizer, args=args,
        temperature=0.7, top_p=0.9
    )
    arr = parse_speed_curvature_text(raw)
    return arr, scene_desc, obj_desc, intent_desc


def predict_with_sc(obs_images, obs_velocities, obs_curvatures,
                    processor, model, tokenizer, args,
                    sc_samples=5):
    preds = []
    first_scene = first_obj = first_intent = None

    for i in range(sc_samples):
        raw, scene_desc, obj_desc, intent_desc = generate_motion_single(
            obs_images, obs_velocities, obs_curvatures,
            processor=processor, model=model, tokenizer=tokenizer, args=args,
            temperature=1.0, top_p=0.9
        )
        arr = parse_speed_curvature_text(raw)
        if arr is not None:
            preds.append(arr)
            if first_scene is None:
                first_scene, first_obj, first_intent = scene_desc, obj_desc, intent_desc

    if not preds:
        return None, None, None, None

    mean_pred = average_predictions(preds)
    return mean_pred, first_scene, first_obj, first_intent


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

    for intent_hint in TOT_CANDIDATE_INTENTS:
        extra_text = f"Assume the driver intent is: {intent_hint}."
        raw, scene_desc, obj_desc, intent_desc = generate_motion_single(
            obs_images, obs_velocities, obs_curvatures,
            processor=processor, model=model, tokenizer=tokenizer, args=args,
            extra_intent_text=extra_text,
            temperature=0.9, top_p=0.9
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
        print(best_pred)
    if best_pred is None:
        return None, None, None, None
    return best_pred, best_scene, best_obj, best_intent


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
    excel_rows = []#excel容器
    all_cam_images_sequence = []
   
    for scene in scenes:
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

        curr_sample_token = first_sample_token
        while True:
            sample = nusc.get("sample", curr_sample_token)
            cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            img_path = os.path.join(nusc.dataroot, cam_front_data["filename"])
            front_camera_images.append(img_path)

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
                pred_vk, scene_desc, obj_desc, intent_desc = predict_with_cot(
                    obs_images_for_vlm, obs_vel, obs_curv, processor, model, tokenizer, args
                )
            elif args.method == "sc":
                pred_vk, scene_desc, obj_desc, intent_desc = predict_with_sc(
                    obs_images_for_vlm, obs_vel, obs_curv, processor, model, tokenizer, args,
                    sc_samples=args.sc_samples
                )
            elif args.method == "tot":
                pred_vk, scene_desc, obj_desc, intent_desc = predict_with_tot(
                    obs_images_for_vlm, obs_vel, obs_curv, processor, model, tokenizer, args
                )
            else:
                raise ValueError(f"Unknown method: {args.method}")

            if pred_vk is None:
                print(f"   → frame {i}: parse failed, skip.")
                continue

            # 曲率缩放回真实值
            pred_curv = pred_vk[:, 1] / 100.0
            pred_speed = pred_vk[:, 0]
            #print预测
            pred_len = min(FUT_LEN, len(pred_curv))
            pred_pairs_str = ",".join([f"[{v:.2f},{k*100:.2f}]" for v,k in zip(pred_speed[:pred_len], pred_curv[:pred_len])])
            obs_pairs_str = ",".join([f"[{np.linalg.norm(v):.2f},{c*100:.2f}]" for v,c in zip(obs_vel, obs_curv)])
        
            fut_traj_world_np = np.array(fut_traj_world)

            # 轨迹积分
            pred_traj = np.zeros((pred_len, 3), dtype=np.float32)
            pred_traj[:, :2] = IntegrateCurvatureForPoints(
                pred_curv[:pred_len],
                pred_speed[:pred_len],
                fut_traj_world_np[0],
                atan2(ego_velocities[i + OBS_LEN - 1][1],
                      ego_velocities[i + OBS_LEN - 1][0]),
                pred_len,
            )

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
                    last_obs_k = obs_curv[-1]*100
                    f.write("First-step difference:\n")
                    f.write(f"delta v = {abs(pred_speed[0] - last_obs_v):.3f}\n")
                    f.write(f"delta k = {abs(pred_curv[0] - last_obs_k):.3f}\n\n")

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
