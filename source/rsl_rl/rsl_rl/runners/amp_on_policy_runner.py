# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.utils import resolve_obs_groups

from rsl_rl.utils import AMPLoader
from rsl_rl.algorithms import AMP_PPO
from rsl_rl.networks import Discriminator, ActorCriticMoE
from rsl_rl.utils import export_policy_as_onnx
from rsl_rl.utils.logger import Logger

class AMPOnPolicyRunner:
    """
    AMPOnPolicyRunner 是一个高级协调器，用于管理使用对抗运动先验 (AMP) 
    结合在线强化学习 (PPO) 的策略训练和评估。

    它整合了多个组件：
    - 环境 (`VecEnv`)
    - 策略 (`ActorCritic`, `ActorCriticRecurrent`)
    - 判别器 (Discriminator)
    - 专家数据集 (AMPLoader)
    - 奖励组合 (任务奖励 + 风格奖励)
    - 日志记录和检查点保存

    ---
    🔧 配置 Configuration
    ----------------
    该类期望一个 `train_cfg` 字典，其结构包含以下键：
    - "obs_groups": 可选映射，描述哪些观测张量属于 "policy" 输入，哪些属于 "critic" 输入。
    - "policy": 策略网络的配置，包括 `"class_name"`。
    - "algorithm": PPO/AMP_PPO 的配置，包括 `"class_name"`。
    - "discriminator": AMP 判别器的配置。
    - "dataset": 传递给 `AMPLoader` 的字典，至少包含：
        * "amp_data_path": 存放 `.npy` 专家数据集的文件夹路径。
        * "datasets": 数据集名称 -> 采样权重 (float) 的映射。
        * "slow_down_factor": 应用于真实运动数据的减速因子，以匹配仿真动力学。
    - "num_steps_per_env": 每个环境的 Rollout 视界长度 (horizon)。
    - "save_interval": 模型检查点保存频率（以迭代次数计）。
    - "empirical_normalization": (已弃用) 镜像到 `policy.actor_obs_normalization` 的旧标志。
    - "logger": "tensorboard", "wandb", 或 "neptune" 之一。

    ---
    📦 数据集格式 Dataset format
    ------------------
    通过 `AMPLoader` 加载的专家运动数据集必须是包含字典的 `.npy` 文件：

    - `"joints_list"`: List[str] — 有序的关节名称列表。
    - `"joint_positions"`: List[np.ndarray] — 每个时间步的关节配置 (1D 数组)。
    - `"root_position"`: List[np.ndarray] — 世界坐标系下的基座位置。
    - `"root_quaternion"`: List[np.ndarray] — **`xyzw`** 格式的基座方向 (SciPy 默认)。
    - `"fps"`: float — 原始数据集帧率。

    内部处理：
    - 四元数通过 SLERP 插值并在使用前转换为 **`wxyz`** 格式 (以匹配 Isaac Gym 惯例)。
    - 速度通过有限差分估算。
    - 所有数据转换为 torch tensors 并放置在指定设备上。

    ---
    🎓 AMP 奖励 AMP Reward
    -------------
    在每个训练步骤中，runner 收集 AMP 特定的观测，并从专家数据集中计算
    基于判别器的“风格奖励”。该奖励与环境奖励结合如下：

        `reward = 0.5 * task_reward + 0.5 * style_reward`
    
    (注意：代码实际实现中通常是在 Env 或 Wrapper 里做混合，或者像这里一样在 Runner 的 step 循环里混合)

    ---
    🔁 训练循环 Training loop
    ----------------
    `learn()` 方法执行以下操作：
    - `rollout`: 通过 `self.alg.act()` 和 `env.step()` 收集 TensorDict 观测。
    - `style_reward`: 通过判别器 `predict_reward(...)` 计算。
    - `storage update`: 通过 `process_env_step()` 和 `process_amp_step()` 更新存储。
    - `return computation`: 使用最新的 TensorDict 观测通过 `compute_returns()` 计算回报。
    - `update`: 使用 `self.alg.update()` 执行反向传播。
    - 通过 TensorBoard/WandB/Neptune 记录日志。

    ---
    💾 保存和 ONNX 导出 Saving and ONNX export
    --------------------------
    在每个 `save_interval`，runner 会：
    - 保存完整状态 (`model`, `optimizer`, `discriminator`, 等)。
    - 可选地将策略导出为 ONNX 模型用于部署。
    - 如果启用，将检查点上传到日志服务。

    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        # 初始化配置
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.discriminator_cfg = train_cfg["discriminator"]
        self.dataset_cfg = train_cfg["dataset"]
        self.device = device
        self.env = env

        # 获取环境的初始观测（用于确定维度）
        observations = self.env.get_observations()
        default_sets = ["critic"]
        # 解析观测分组 (Policy 输入 vs Critic 输入)
        self.cfg["obs_groups"] = resolve_obs_groups(
            observations, self.cfg.get("obs_groups"), default_sets
        )

        self.alg: AMP_PPO = self._construct_algorithm(observations)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        # 初始化 Storage (Rollout Buffer)
        obs_template = observations.clone().detach().to(self.device)
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_template,
            (self.env.num_actions,),
        )

        # 日志相关初始化
        self.log_dir = log_dir
        self.writer = None
        self.logger_type = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """执行主训练循环。"""
        
        # 初始化日志记录器 (Writer)
        if self.log_dir is not None and self.writer is None:
            # 默认使用 Tensorboard，也支持 Neptune 和 WandB
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                # Neptune Logger 初始化
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "wandb":
                # WandB Logger 初始化
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                import wandb

                # 一个辅助函数，用于给 run name 加上自动递增的序号后缀 (模拟 rsl-rl 旧版行为)
                def update_run_name_with_sequence(prefix: str) -> None:
                    project = wandb.run.project
                    entity = wandb.run.entity
                    api = wandb.Api()
                    runs = api.runs(f"{entity}/{project}")
                    max_num = 0
                    for run in runs:
                        if run.name.startswith(prefix):
                            numeric_suffix = run.name[len(prefix) :]
                            try:
                                run_num = int(numeric_suffix)
                                if run_num > max_num:
                                    max_num = run_num
                            except ValueError:
                                continue
                    new_num = max_num + 1
                    new_run_name = f"{prefix}{new_num}"
                    wandb.run.name = new_run_name
                    print("Updated run name to:", wandb.run.name)

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                update_run_name_with_sequence(prefix=self.cfg["wandb_project"])
                wandb.gym.monitor()
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                # Tensorboard Logger 初始化
                self.writer = TensorboardSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10
                )
            else:
                raise AssertionError("logger type not found")

        # 随机化初始回合长度 (防止所有环境同步 Reset)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
            
        # 获取初始观测
        obs = self.env.get_observations().to(self.device)
        # 获取初始 AMP 观测 (用于计算 reward, 需要上一帧和当前帧)
        amp_obs = obs["amp"].clone()
        
        self.train_mode()  # 切换到训练模式

        # 统计数据缓存
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        # >>> 主循环开始 <<<
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # --- 1. Rollout (数据收集) ---

            mean_style_reward_log = 0
            mean_task_reward_log = 0

            with torch.inference_mode(): # 不计算梯度
                for _ in range(self.num_steps_per_env): # 每次 rollout 收集 N 步
                    # 1. 策略网络输出动作
                    actions = self.alg.act(obs)
                    # 记录当前 AMP 观测 (state t)
                    self.alg.act_amp(amp_obs)
                    
                    # 2. 环境步进
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # 获取下一帧 AMP 观测 (state t+1)
                    next_amp_obs = obs["amp"].clone()
                    
                    # 3. 计算风格奖励 (Style Reward)
                    # 输入是 (s_t, s_{t+1}) 对
                    style_rewards = self.discriminator.predict_reward(
                        amp_obs, next_amp_obs
                    )

                    # 记录奖励统计
                    mean_task_reward_log += rewards.mean().item()
                    mean_style_reward_log += style_rewards.mean().item()

                    # 4. 混合奖励: Task Reward + Style Reward
                    # 系数 0.5 是硬编码的，可以提取到 config 中
                    rewards = 0.5 * rewards + 0.5 * style_rewards

                    # 5. 处理数据并存入 PPO Storage
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # 存入 AMP Replay Buffer
                    self.alg.process_amp_step(next_amp_obs)

                    # 更新 amp_obs 为当前帧，供下一步使用
                    amp_obs = next_amp_obs

                    # 6. 处理日志信息 (Episode 结束时的统计)
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        
                        # 处理 Done 的环境
                        new_ids = torch.nonzero(dones, as_tuple=False)
                        if new_ids.numel() > 0:
                            env_indices = new_ids.view(-1)
                            rewbuffer.extend(cur_reward_sum[env_indices].cpu().tolist())
                            lenbuffer.extend(
                                cur_episode_length[env_indices].cpu().tolist()
                            )
                            # 重置统计器
                            cur_reward_sum[env_indices] = 0
                            cur_episode_length[env_indices] = 0

                stop = time.time()
                collection_time = stop - start

                # --- 2. Learning Step (学习步骤) ---
                start = stop
                # 计算 GAE 回报
                self.alg.compute_returns(obs)

            # 归一化 log 数据
            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env

            # 执行 PPO + AMP 更新
            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_accuracy_policy,
                mean_accuracy_expert,
                mean_kl_divergence,
            ) = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # --- 3. Logging & Saving (日志与保存) ---
            if self.log_dir is not None:
                self.log(locals()) # 打印和写入 TensorBoard
            
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), save_onnx=True)
            
            ep_infos.clear()
            
            # 保存 git 状态 (仅第一次迭代)
            if it == start_iter:
                pass
                # git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                #     for path in git_file_paths:
                #         self.writer.save_file(path)

        # 训练结束，保存最终模型
        self.save(
            os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
            save_onnx=True,
        )

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """处理日志打印到控制台和写入 Logger。"""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        # 处理 Env 返回的额外信息 (extras['episode'])
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info: continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                
                value = torch.mean(infotensor)
                # 写入 Writer
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        # 获取 Action Noise Std 用于监控探索程度
        if getattr(self.alg.actor_critic, "noise_std_type", "scalar") == "log":
            mean_std_value = torch.exp(self.alg.actor_critic.log_std).mean()
        else:
            mean_std_value = self.alg.actor_critic.std.mean()
            
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        # 写入标准 PPO 损失
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])

        # 写入 AMP 相关指标 (Loss, Prediction, Accuracy)
        self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar("Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"])
        self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
        self.writer.add_scalar("Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"])
        self.writer.add_scalar("Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"])

        # 写入学习率和 KL 散度
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Loss/mean_kl_divergence", locs["mean_kl_divergence"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std_value.item(), locs["it"])
        
        # 写入性能指标 (FPS, Time)
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        
        # 写入训练奖励 (Reward Buffer 统计)
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"])
            self.writer.add_scalar("Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"])
            
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        # 构建打印字符串
        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        
        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std_value.item():.2f}\n"""
        )
        
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )

        log_string += ep_string

        # 计算 ETA
        eta_seconds = (
            self.tot_time
            / (locs["it"] + 1)
            * (locs["num_learning_iterations"] - locs["it"])
        )
        eta_h, rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(rem, 60)

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None, save_onnx=False):
        """保存模型检查点。"""
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

        if save_onnx:
            # 导出 ONNX
            onnx_folder = os.path.dirname(path)
            iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
            onnx_model_name = f"policy_{iteration}.onnx"

            export_policy_as_onnx(
                self.alg.actor_critic,
                normalizer=self.alg.actor_critic.actor_obs_normalizer,
                path=onnx_folder,
                filename=onnx_model_name,
            )

            if self.logger_type in ["neptune", "wandb"]:
                self.writer.save_model(
                    os.path.join(onnx_folder, onnx_model_name),
                    self.current_learning_iteration,
                )

    def load(self, path, load_optimizer=True, weights_only=False):
        """加载模型检查点。"""
        loaded_dict = torch.load(
            path, map_location=self.device, weights_only=weights_only
        )
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        
        # 加载判别器
        discriminator_state = loaded_dict["discriminator_state_dict"]
        self.alg.discriminator.load_state_dict(discriminator_state, strict=False)

        # 兼容旧版本：加载分离的 Normalizer
        amp_normalizer_module = loaded_dict.get("amp_normalizer")
        if amp_normalizer_module is not None and getattr(
            self.alg.discriminator, "empirical_normalization", False
        ):
            self.alg.discriminator.amp_normalizer.load_state_dict(
                amp_normalizer_module.state_dict()
            )
            
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """获取用于推理的策略函数。"""
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def train_mode(self):
        """设置全模块为训练模式。"""
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

    def eval_mode(self):
        """设置全模块为评估模式。"""
        self.alg.actor_critic.eval()
        self.alg.discriminator.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)
    
    def _construct_algorithm(self, observations: TensorDict) -> AMP_PPO:
                # Create the algorithm
        # 动态加载并实例化 Policy 类 (ActorCritic, ActorCriticRecurrent, MoE)
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # e.g., ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticMoE = (
            actor_critic_class(
                observations,
                self.cfg["obs_groups"],
                self.env.num_actions,
                **self.policy_cfg,
            ).to(self.device)
        )
        
        # NOTE: 为了使用 AMP，我们需要确保环境的观测配置与 AMP 观测一致。
        # 这里从 Isaac Lab 的配置中提取 AMP 关节名称。
        # 注意：这行代码强依赖于 Manager-Based 环境结构。
        # amp_joint_names = self.env.cfg.observations.amp.joint_pos.params[
        #     "asset_cfg"
        # ].joint_names

        # --- [修改开始] 兼容 Direct 和 Manager Based 环境 ---
        
        # 尝试方法 1: Direct 环境通常在 cfg 中直接存有关节名称列表 (取决于你怎么写的 Config)
        if hasattr(self.env.unwrapped, "cfg") and hasattr(self.env.unwrapped.cfg, "dof_names"):
             amp_joint_names = self.env.unwrapped.cfg.dof_names
        
        # 尝试方法 2: Direct 环境已经初始化了 robot，可以直接从 robot 实例获取
        elif hasattr(self.env.unwrapped, "robot") and hasattr(self.env.unwrapped.robot.data, "joint_names"):
            # 注意：DirectRLEnv 的 robot.data.joint_names 通常是所有关节
            # 如果 AMP 只需要一部分，你需要手动指定。
            # 这里假设我们需要所有驱动关节
            amp_joint_names = self.env.unwrapped.robot.data.joint_names
            
        # 尝试方法 3: Manager Based 环境 (原有的逻辑)
        elif hasattr(self.env.cfg, "observations") and hasattr(self.env.cfg.observations, "amp"):
            amp_joint_names = self.env.cfg.observations.amp.joint_pos.params["asset_cfg"].joint_names
            
        # 尝试方法 4: 硬编码回退 (实在不行，就在 Config 里加一个字段)
        elif "amp_joint_names" in self.dataset_cfg:
             amp_joint_names = self.dataset_cfg["amp_joint_names"]
             
        else:
            raise AttributeError(
                "Could not find joint names for AMPLoader. \n"
                "If using DirectRLEnv, please ensure 'dof_names' is in your env cfg or 'amp_joint_names' is in your train_cfg['dataset'].\n"
                "If using ManagerBasedRLEnv, ensure cfg.observations.amp structure exists."
            )

        # 初始化 AMP 所需的所有配料 (判别器，数据集加载器)
        
        # 尝试从 Environment 实例直接读取定义好的尺寸
        if hasattr(self.env.unwrapped, "amp_observation_size"):
            num_amp_obs = self.env.unwrapped.amp_observation_size
        else:
            raise AttributeError("Cannot determine AMP observation size. Please define 'self.amp_observation_size' in your Env.")
        
        # 初始化 AMPLoader：负责加载和采样专家动作
        amp_data = AMPLoader(
            device=self.device,
            dataset_path_root=self.dataset_cfg["amp_data_path"],
            datasets=self.dataset_cfg["datasets"],
            simulation_dt=self.env.cfg.sim.dt * self.env.cfg.decimation, # 物理仿真步长
            slow_down_factor=self.dataset_cfg["slow_down_factor"],
            expected_joint_names=amp_joint_names,
        )

        # 初始化 Discriminator：判别真假动作
        self.discriminator = Discriminator(
            input_dim=num_amp_obs * 2,  # 判别器输入是 (当前观测 + 下一观测) 的拼接，所以维度乘 2
            hidden_layer_sizes=self.discriminator_cfg["hidden_dims"],
            reward_scale=self.discriminator_cfg["reward_scale"],
            device=self.device,
            loss_type=self.discriminator_cfg["loss_type"],
            empirical_normalization=self.discriminator_cfg["empirical_normalization"],
        ).to(self.device)

        # 初始化 PPO 算法
        alg_class = eval(self.alg_cfg.pop("class_name"))  # e.g., AMP_PPO
        
        # 清理 alg_cfg：移除那些在 AMP_PPO 中不存在但在标准 rsl_rl PPO 中存在的参数，防止报错
        # (例如 RND, Symmetry, Multi-GPU config 等)
        for key in list(self.alg_cfg.keys()):
            if key not in AMP_PPO.__init__.__code__.co_varnames:
                self.alg_cfg.pop(key)

        # 实例化 AMP_PPO 算法
        alg: AMP_PPO = alg_class(
            actor_critic=actor_critic,
            discriminator=self.discriminator,
            amp_data=amp_data,
            device=self.device,
            **self.alg_cfg,
        )
        return alg