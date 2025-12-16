#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating the use of CoT-SC and ToT reasoning methods
in the OpenEMMA framework.
"""

import argparse
import os
import sys

# Add the parent directory to the path to import openemma modules
# [注释] 添加父目录到路径中以便导入openemma模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# [注释] 导入基础OpenEMMA类
from vlm.base_backbone import BaseOpenEMMA


# [注释] 定义参数类，用于配置模型参数
class Args:
    def __init__(self, model_id="gpt-4", api_key=None, reasoning_mode="cot"):
        self.model_id = model_id
        self.api_key = api_key
        self.reasoning_mode = reasoning_mode


# [注释] 主函数，演示如何使用不同的推理模式
def main():
    # [注释] 解析命令行参数
    parser = argparse.ArgumentParser(description="OpenEMMA Reasoning Example")
    parser.add_argument("--model_id", type=str, default="gpt-4", 
                        help="Model ID to use for reasoning")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for the model (if required)")
    parser.add_argument("--reasoning_mode", type=str, default="cot",
                        choices=["cot", "cot-sc", "tot"],
                        help="Reasoning mode: cot, cot-sc, or tot")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    
    args = parser.parse_args()
    
    # [注释] 创建模型参数对象
    model_args = Args(model_id=args.model_id, api_key=args.api_key, 
                      reasoning_mode=args.reasoning_mode)
    
    # [注释] 初始化OpenEMMA模型
    model = BaseOpenEMMA(model_args)
    
    # [注释] 示例数据（实际使用中会从数据集中获取）
    sample_data = {
        "gt_ego_fut_diff": [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]],
        "gt_ego_fut_trajs": [[1.0, 2.0], [1.5, 3.0], [2.0, 4.0], [2.5, 5.0], [3.0, 6.0]],
        "gt_ego_his_diff": [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
        "gt_ego_his_trajs": [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0], [1.5, 3.0], [2.0, 4.0]]
    }
    
    # [注释] 使用指定的推理模式生成航路点
    command = "MOVE FORWARD"
    waypoints = model.generate_waypoints(
        command=command,
        image_path=args.image_path,
        data=sample_data,
        backbone=model,
        args=model_args
    )
    
    # [注释] 输出结果
    print(f"Reasoning mode: {args.reasoning_mode}")
    print(f"Generated waypoints: {waypoints}")


# [注释] 程序入口点
if __name__ == "__main__":
    main()
