# Copyright (c) 2022-2025, The Isaac Lab Project Developers ...
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.
此脚本用于使用 skrl 库训练强化学习智能体。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

# AppLauncher: Isaac Lab 启动仿真器的核心工具
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# 1. 定义命令行参数解析器
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")

# --video: 是否录制训练过程的视频
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# --video_length: 录制视频的长度（步数），默认 200 步
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# --video_interval: 录制间隔，每隔多少步录一次
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")

# --num_envs: 并行环境数量，覆盖配置文件
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# --task: 任务名称（例如 "Isaac-Cartpole-v0"），必填
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# --agent: 算法配置入口点名称（例如 "skrl_ppo_cfg"）
# 如果不填，脚本会根据 --algorithm 参数自动推断
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)

# --seed: 随机种子
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# --distributed: 分布式训练开关
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# --checkpoint: 检查点路径，用于恢复训练 (Resume)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
# --max_iterations: 最大训练迭代次数
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# --export_io_descriptors: 导出 IO 描述符（高级功能）
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")

# --ml_framework: [SKRL特有] 选择深度学习后端
# "torch": 使用 PyTorch (默认)
# "jax": 使用 JAX
# "jax-numpy": 使用 JAX 的 NumPy 模拟后端
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)

# --algorithm: [SKRL特有] 选择强化学习算法
# 默认为 PPO，但也支持 AMP, IPPO, MAPPO 等
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# Ray 集成参数，通常不用管
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)

# 添加 AppLauncher 参数 (如 --headless)
AppLauncher.add_app_launcher_args(parser)

# 解析参数
args_cli, hydra_args = parser.parse_known_args()

# 如果录制视频，强制开启相机
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 sys.argv 给 Hydra 使用
sys.argv = [sys.argv[0]] + hydra_args

# -----------------------------------------------------------------------------
# 2. 启动 Omniverse 仿真器
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import random
import time
from datetime import datetime

import skrl
from packaging import version

# -----------------------------------------------------------------------------
# 3. 检查 SKRL 版本
# -----------------------------------------------------------------------------
SKRL_VERSION = "1.4.3" # 最低支持版本
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

# -----------------------------------------------------------------------------
# 4. 根据框架选择 Runner
# -----------------------------------------------------------------------------
# skrl 的 Runner 负责管理 Agent 和 Environment 之间的交互循环
if args_cli.ml_framework.startswith("torch"):
    # 如果选 PyTorch，导入 PyTorch 版 Runner
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    # 如果选 JAX，导入 JAX 版 Runner
    from skrl.utils.runner.jax import Runner

# 导入 Isaac Lab 环境类
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# 工具函数：解析文件路径、打印字典、导出 YAML
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

# [关键] 导入 SKRL 的环境包装器
# 这个 wrapper 负责把 Isaac Lab 的环境转成 SKRL 能读懂的格式
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 5. 推断配置入口点 (Config Entry Point)
# -----------------------------------------------------------------------------
# 如果用户没有显式指定 --agent 参数，我们根据 --algorithm 参数猜一个名字
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    # 如果是 PPO，名字叫 "skrl_cfg_entry_point"
    # 如果是 AMP，名字叫 "skrl_amp_cfg_entry_point"
    # 这些字符串必须在任务的 __init__.py 的 kwargs 里注册过！
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    # 如果用户指定了 agent，就用用户的，并解析出算法名
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# 使用 hydra 装饰器加载环境配置 (env_cfg) 和算法配置 (agent_cfg)
# 注意：skrl 的 agent_cfg 是一个纯 Python 字典 (dict)，而不是类实例
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    
    # -------------------------------------------------------------------------
    # 1. 配置覆盖 (Configuration Override)
    # -------------------------------------------------------------------------
    # 使用 CLI 参数覆盖环境配置
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 检查：分布式训练不支持 CPU
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # -------------------------------------------------------------------------
    # 2. 分布式与多卡设置
    # -------------------------------------------------------------------------
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    
    # 覆盖训练迭代次数
    # skrl 的配置里用 timesteps = max_iterations * num_envs (总步数)
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    
    # 设置程序退出时不自动关闭环境（由脚本最后手动关闭）
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    
    # 设置 SKRL 的 JAX 后端 (如果用了 JAX)
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # -------------------------------------------------------------------------
    # 3. 随机种子设置
    # -------------------------------------------------------------------------
    # 如果种子是 -1，随机生成一个
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # 同步种子到 Agent 配置和 Env 配置
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # -------------------------------------------------------------------------
    # 4. 日志路径设置 (Logging)
    # -------------------------------------------------------------------------
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # 生成实验名：时间戳_算法名_框架名 (如 2024-01-01_12-00_ppo_torch)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line: {log_dir}")
    
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
        
    # 把路径写回 agent_cfg 字典，因为 SKRL 内部 Runner 会用这个路径
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # 导出配置为 YAML
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # -------------------------------------------------------------------------
    # 5. 检查点与环境创建
    # -------------------------------------------------------------------------
    # 获取恢复训练的模型路径
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # Manager-Based 环境导出 IO 描述
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager based RL environments...")

    env_cfg.log_dir = log_dir

    # 创建 Gym 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 单智能体转换：如果是 PPO 跑在多智能体环境上，强制转为单智能体接口
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # 视频录制 Wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # -------------------------------------------------------------------------
    # 6. SKRL 环境包装与训练 (SKRL Wrapper & Runner)
    # -------------------------------------------------------------------------
    # 使用 SkrlVecEnvWrapper 包装环境
    # 这个 wrapper 极其重要，它把 Isaac Lab 的 Tensor 转换为 SKRL 需要的格式
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # 实例化 SKRL Runner
    # SKRL 的 Runner 设计非常高度集成，它会：
    # 1. 根据 agent_cfg["agent"] 里的配置自动创建模型 (MLP/RNN/CNN)
    # 2. 创建 Agent (PPO/AMP...)
    # 3. 创建 Trainer
    runner = Runner(env, agent_cfg)

    # 如果需要恢复训练，加载模型权重
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # 🚀 开始训练循环
    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()