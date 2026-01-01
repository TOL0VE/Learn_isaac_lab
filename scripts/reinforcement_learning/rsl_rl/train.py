# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

# AppLauncher 是 Isaac Lab 用来启动 Omniverse 仿真器核心的工具类
from isaaclab.app import AppLauncher

# local imports
# cli_args 是本地的一个辅助模块，用来处理跟 RSL-RL 相关的特定参数
import cli_args  # isort: skip

# -----------------------------------------------------------------------------
# 1. 定义命令行参数解析器
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")

# --video: 开关，是否在训练过程中录制视频
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# --video_length: 录制的视频长度（步数），默认录 200 步
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# --video_interval: 每隔多少步录一次视频，默认 2000 步
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")

# --num_envs: 并行环境的数量。如果指定了，会覆盖配置文件里的默认值。
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# --task: 任务名称（例如 "Isaac-Cartpole-v0"），这是必须指定的，用来去 Gym 注册表里找环境
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# --agent: 算法配置的入口点名称，默认为 "rsl_rl_cfg_entry_point"
# 这对应了 gym.register 时 kwargs 里注册的那个 key
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
# --seed: 随机种子，用于复现实验结果
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# --max_iterations: 训练的最大迭代次数（PPO 的大循环次数）
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# --distributed: 是否开启多 GPU 分布式训练
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# --export_io_descriptors: 是否导出 I/O 描述文件（用于 Warp 等高级特性，一般不用）
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# --ray-proc-id: Ray 框架使用的进程 ID，如果不用 Ray 调参可以忽略
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)

# 添加 RSL-RL 特有的参数（比如 --resume, --run_name 等）
cli_args.add_rsl_rl_args(parser)

# 添加 AppLauncher 特有的参数（比如 --headless, --device 等）
AppLauncher.add_app_launcher_args(parser)

# 解析参数
# parse_known_args 允许传入一些 parser 不认识的参数（这些多余的参数会被 hydra 捡走）
args_cli, hydra_args = parser.parse_known_args()

# 如果开启了视频录制，强制开启相机功能
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 sys.argv，只保留脚本名和 hydra 需要的参数
# 这是为了防止 argparse 处理过的参数干扰后续的 Hydra 配置加载
sys.argv = [sys.argv[0]] + hydra_args

# -----------------------------------------------------------------------------
# 2. 启动 Omniverse 仿真器
# -----------------------------------------------------------------------------
# 初始化 AppLauncher，这会读取 --headless 等参数并配置 Kit
app_launcher = AppLauncher(args_cli)
# 真正的启动！这一行执行后，Isaac Sim 的核心才被加载，才能 import pxr/omni 等库
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# 定义所需的最低 rsl-rl 版本
RSL_RL_VERSION = "3.0.1"
# 获取当前环境安装的版本
installed_version = metadata.version("rsl-rl-lib")

# 如果版本太低，打印报错信息并退出
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)


    """Rest everything follows."""

import gymnasium as gym
import logging
import os
import time
import torch
from datetime import datetime

# 导入 RSL-RL 的 Runner (执行训练循环的核心类)
# OnPolicyRunner 用于 PPO，DistillationRunner 用于蒸馏
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# 导入 Isaac Lab 的环境相关类
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent, # 辅助函数：把多智能体环境包装成单智能体接口
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

# 导入 Isaac Lab 对 RSL-RL 的适配器
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

# 导入 isaaclab_tasks 以便注册所有任务 (gym.make 能找到它们)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
# 这是一个装饰器，用于从注册表加载 Hydra 配置
from isaaclab_tasks.utils.hydra import hydra_task_config

# 初始化日志记录器
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

# -----------------------------------------------------------------------------
# PyTorch 性能优化设置
# -----------------------------------------------------------------------------
# 允许使用 TF32 格式（在 Ampere 架构 GPU 上加速矩阵乘法）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 关闭确定性计算（为了速度，如果需要完全复现性应设为 True）
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# 装饰器作用：
# 1. 拦截 main 调用
# 2. 根据 args_cli.task 找到任务配置 -> env_cfg
# 3. 根据 args_cli.task 和 args_cli.agent 找到算法配置 -> agent_cfg
# 4. 把这两个对象传给 main 函数
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    
    # -------------------------------------------------------------------------
    # 1. 配置覆盖 (Configuration Override)
    # -------------------------------------------------------------------------
    # 使用命令行参数覆盖配置文件里的设置（比如你命令行敲了 --seed 42，就要覆盖 yaml 里的 seed）
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    # 如果命令行指定了 --num_envs，覆盖环境配置
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # 如果命令行指定了 --max_iterations，覆盖算法配置
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # -------------------------------------------------------------------------
    # 2. 随机种子与设备设置 (Seed & Device)
    # -------------------------------------------------------------------------
    # 同步环境和算法的种子
    env_cfg.seed = agent_cfg.seed
    # 设置运行设备 (cuda:0 或 cpu)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # 检查：分布式训练不支持 CPU
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # -------------------------------------------------------------------------
    # 3. 分布式训练设置 (Multi-GPU)
    # -------------------------------------------------------------------------
    if args_cli.distributed:
        # 根据本地 rank 设置设备（比如进程 0 用 cuda:0，进程 1 用 cuda:1）
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 调整种子：不同进程必须用不同的种子，否则所有 GPU 采样的经验都一样，训练就废了
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # -------------------------------------------------------------------------
    # 4. 日志目录设置 (Logging)
    # -------------------------------------------------------------------------
    # 根目录：logs/rsl_rl/实验名
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # 子目录：时间戳_运行名 (例如 2024-01-01_12-00-00_run1)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # -------------------------------------------------------------------------
    # 5. I/O 描述符导出 (对于 Manager-Based 环境)
    # -------------------------------------------------------------------------
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        # Direct 模式不支持这个（也不需要）
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # 将计算好的日志目录塞回环境配置里
    env_cfg.log_dir = log_dir

    # -------------------------------------------------------------------------
    # 6. 创建环境 (Create Environment)
    # -------------------------------------------------------------------------
    # gym.make 会调用 isaaclab.envs:ManagerBasedRLEnv 或 DirectRLEnv
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 特殊处理：如果底层是多智能体环境 (MARL)，但我们要用单智能体算法跑 (PPO)
    # 这个 wrapper 会把所有智能体的观测拼起来
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # -------------------------------------------------------------------------
    # 7. 检查点恢复路径 (Checkpoint Resume)
    # -------------------------------------------------------------------------
    # 如果指定了 --resume，计算之前的模型路径
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # -------------------------------------------------------------------------
    # 8. 视频录制包装器 (Video Recording Wrapper)
    # -------------------------------------------------------------------------
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0, # 每隔 interval 步录一次
            "video_length": args_cli.video_length, # 录多长
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        # 使用 Gym 标准的 RecordVideo wrapper
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # -------------------------------------------------------------------------
    # 9. RSL-RL 环境包装器 (RSL-RL Wrapper)
    # -------------------------------------------------------------------------
    # RSL-RL 期望环境返回的是 torch.Tensor，并且位于 GPU 上
    # RslRlVecEnvWrapper 负责把 Gym 的接口转换成 RSL-RL 喜欢的接口
    # 并且处理 clip_actions (把网络输出裁剪到 [-1, 1])
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -------------------------------------------------------------------------
    # 10. 创建 Runner 并开始训练 (Create Runner & Learn)
    # -------------------------------------------------------------------------
    # 根据配置类名选择 Runner
    if agent_cfg.class_name == "OnPolicyRunner":
        # 标准 PPO 训练器
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        # 蒸馏训练器
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    # 记录当前代码的 git hash 到日志，方便以后查版本
    runner.add_git_repo_to_log(__file__)
    
    # 如果是 Resume 模式，加载模型权重
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # 把最终生效的参数保存为 YAML，方便复查
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 🚀 开始训练！
    # init_at_random_ep_len=True: 训练开始时，随机化每个环境的当前步数
    # 这可以防止所有环境同时 Reset，导致数据分布出现周期性波动
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # -------------------------------------------------------------------------
    # 11. 清理工作 (Cleanup)
    # -------------------------------------------------------------------------
    # 关闭环境和仿真器
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()