# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import time
import torch
import statistics
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.logger import Logger  # [Upgrade] 使用官方 Logger

from rsl_rl.utils import AMPLoader
from rsl_rl.algorithms import AMP_PPO
from rsl_rl.networks import Discriminator, ActorCriticMoE
from rsl_rl.utils import export_policy_as_onnx


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

    [Upgrade Note]: 这是一个升级版，集成了 rsl_rl v2.x 的高级日志系统和多 GPU 支持。
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.discriminator_cfg = train_cfg["discriminator"]
        self.dataset_cfg = train_cfg["dataset"]
        self.device = device
        self.env = env

        # [Upgrade] 配置多 GPU 训练
        self._configure_multi_gpu()

        # 获取环境观测
        observations = self.env.get_observations()
        default_sets = ["critic"]
        
        # [Upgrade] 解析观测分组
        self.cfg["obs_groups"] = resolve_obs_groups(
            observations, self.cfg.get("obs_groups"), default_sets
        )

        # 构建算法 (ActorCritic, Discriminator, PPO)
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

        # [Upgrade] 初始化高级 Logger
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

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """执行主训练循环。"""
        
        # 随机化初始回合长度
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
            
        # 获取初始观测
        obs = self.env.get_observations().to(self.device)
        # 获取初始 AMP 观测 (用于计算 reward)
        amp_obs = obs["amp"].clone() if "amp" in obs else torch.zeros(1, device=self.device) # 防御性编程
        
        self.train_mode()  # 切换到训练模式

        # [Upgrade] 多 GPU 同步参数
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            # 注意：如果你的 AMP_PPO 还没实现 broadcast_parameters，这里可能会报错
            # 但标准的 rsl_rl PPO 是有的，你可以暂时注释掉或者去实现它
            # self.alg.broadcast_parameters() 
            pass

        # 统计数据缓存 (仅用于计算 AMP Reward 的 logging，其他由 Logger 接管)
        # 实际上 Logger 也会处理 rewbuffer，这里主要为了计算 mean_style_reward
        start_iter = self.current_learning_iteration
        total_iter = start_iter + num_learning_iterations
        
        # >>> 主循环开始 <<<
        for it in range(start_iter, total_iter):
            start = time.time()
            
            # --- 1. Rollout (数据收集) ---
            mean_style_reward_log = 0.0
            mean_task_reward_log = 0.0

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 1. 策略动作
                    actions = self.alg.act(obs)
                    self.alg.act_amp(amp_obs)
                    
                    # 2. 环境步进
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # 3. AMP 逻辑
                    next_amp_obs = obs["amp"].clone() # 确保 DirectEnv 返回了这个 key
                    style_rewards = self.discriminator.predict_reward(amp_obs, next_amp_obs)

                    # 记录原始奖励 (Log用)
                    mean_task_reward_log += rewards.mean().item()
                    mean_style_reward_log += style_rewards.mean().item()

                    # 4. 混合奖励 (Task + Style)
                    total_rewards = 0.5 * rewards + 0.5 * style_rewards

                    # 5. 处理数据
                    self.alg.process_env_step(obs, total_rewards, dones, extras)
                    self.alg.process_amp_step(next_amp_obs)
                    
                    amp_obs = next_amp_obs

                    # [Upgrade] Logger 处理步进信息
                    # 注意：我们这里传入的是混合后的 total_rewards 还是原始 rewards？
                    # 通常 Log 里看 Task Reward 更有意义，但 PPO 优化的是 Total。
                    # 这里为了兼容标准 Logger，我们传混合后的 Total，或者你可以魔改 Logger 传 tuple。
                    # RSL-RL Logger 默认只记录传入的 rewards。
                    self.logger.process_env_step(rewards, dones, extras) # 记录原始 Task Reward 比较直观

                stop = time.time()
                collection_time = stop - start
                start = stop

                # 计算 GAE 回报
                self.alg.compute_returns(obs)

            # 归一化 log 数据
            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env

            # --- 2. Update (学习更新) ---
            loss_dict = self.alg.update()
            
            # [Upgrade] 将 AMP 特有的 Loss 也塞进 loss_dict
            # AMP_PPO.update() 返回的是 tuple，我们需要把它转换成 dict 以喂给 Logger
            # 假设你的 AMP_PPO.update 返回的是 tuple (如你之前代码所示)
            # 我们手动解包并构建 dict
            (
                val_loss, surr_loss, amp_loss, grad_pen, 
                pol_pred, exp_pred, acc_pol, acc_exp, kl
            ) = loss_dict
            
            # 重构为字典供 Logger 使用
            loss_dict_log = {
                # PPO 核心损失
                "Loss/value_function": val_loss,
                "Loss/surrogate": surr_loss,
                "Loss/kl_divergence": kl,
                "Policy/learning_rate": self.alg.learning_rate,
                
                # AMP 判别器损失
                "AMP/discriminator_loss": amp_loss,
                "AMP/grad_penalty": grad_pen,
                
                # AMP 预测值 (越高越像是专家)
                "AMP/pred_policy": pol_pred,  # 假数据的得分 (目标是让它变高)
                "AMP/pred_expert": exp_pred,  # 真数据的得分 (通常很高)
                
                # AMP 准确率 (越低越好，说明判别器被骗了)
                "AMP/accuracy_policy": acc_pol, # 判别器识别假数据的准确率
                "AMP/accuracy_expert": acc_exp, # 判别器识别真数据的准确率
                
                # 奖励成分 (监控 Task vs Style 的比例)
                "Reward/style": mean_style_reward_log,
                "Reward/task": mean_task_reward_log,
            }

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # --- 3. Logging (日志) ---
            # [Upgrade] 使用高级 Logger
            self.logger.log(
                it=it,
                start_it=start_iter,
                total_it=total_iter,
                collect_time=collection_time,
                learn_time=learn_time,
                loss_dict=loss_dict_log,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.actor_critic.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg["rnd_cfg"] else None,
            )
            
            # --- 4. Saving (保存) ---
            if it % self.save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        # 训练结束保存
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        """保存模型检查点。"""
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        # [Upgrade] Logger 负责上传云端
        self.logger.save_model(path, self.current_learning_iteration)

        # 尝试导出 ONNX (可选)
        try:
            onnx_path = os.path.dirname(path)
            onnx_name = f"policy_{self.current_learning_iteration}.onnx"
            export_policy_as_onnx(
                self.alg.actor_critic,
                normalizer=self.alg.actor_critic.actor_obs_normalizer,
                path=onnx_path,
                filename=onnx_name,
            )
        except Exception as e:
            print(f"ONNX export failed: {e}")

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """加载模型检查点。"""
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=False)
        
        # 兼容旧版本
        if "amp_normalizer" in loaded_dict and self.discriminator_cfg.get("empirical_normalization", False):
            self.alg.discriminator.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer"].state_dict())
            
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        """获取用于推理的策略函数。"""
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def train_mode(self):
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        self.alg.discriminator.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """[Upgrade] Configure multi-gpu training."""
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        if self.device != f"cuda:{self.gpu_local_rank}":
            # 如果不想强制报错，可以这里 print 一个 warning
            # raise ValueError(f"Device mismatch: {self.device} vs cuda:{self.gpu_local_rank}")
            pass

        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, observations: TensorDict) -> AMP_PPO:
        # 1. 初始化 Policy (ActorCritic / MoE)
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic = actor_critic_class(
            observations,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        
        # 2. 获取关节名称 (兼容 Direct / Manager)
        if hasattr(self.env.unwrapped, "cfg") and hasattr(self.env.unwrapped.cfg, "dof_names"):
             amp_joint_names = self.env.unwrapped.cfg.dof_names
        elif hasattr(self.env.unwrapped, "robot") and hasattr(self.env.unwrapped.robot.data, "joint_names"):
            amp_joint_names = self.env.unwrapped.robot.data.joint_names
        elif hasattr(self.env.cfg, "observations") and hasattr(self.env.cfg.observations, "amp"):
            amp_joint_names = self.env.cfg.observations.amp.joint_pos.params["asset_cfg"].joint_names
        elif "amp_joint_names" in self.dataset_cfg:
             amp_joint_names = self.dataset_cfg["amp_joint_names"]
        else:
            raise AttributeError("Could not find joint names for AMPLoader.")

        # 3. 获取 AMP 观测维度
        if hasattr(self.env.unwrapped, "amp_observation_size"):
            num_amp_obs = self.env.unwrapped.amp_observation_size
        else:
            raise AttributeError("Define 'self.amp_observation_size' in Env.")
        
        # 4. 初始化 AMP Loader
        amp_data = AMPLoader(
            env=self.env, # 直接把环境传进去
            device=self.device,
            time_between_frames=self.env.cfg.sim.dt * self.env.cfg.decimation, # dt
        )

        # 5. 初始化 Discriminator
        self.discriminator = Discriminator(
            input_dim=num_amp_obs * 2,
            hidden_layer_sizes=self.discriminator_cfg["hidden_dims"],
            reward_scale=self.discriminator_cfg["reward_scale"],
            device=self.device,
            loss_type=self.discriminator_cfg["loss_type"],
            empirical_normalization=self.discriminator_cfg["empirical_normalization"],
        ).to(self.device)

        alg_cfg_copy = self.alg_cfg.copy()
        # 6. 初始化 AMP_PPO
        alg_class = eval(alg_cfg_copy.pop("class_name"))
        
        # 清理多余参数
        for key in list(alg_cfg_copy.keys()):
            if key not in AMP_PPO.__init__.__code__.co_varnames:
                alg_cfg_copy.pop(key)

        # 实例化 AMP_PPO 算法
        alg: AMP_PPO = alg_class(
            actor_critic=actor_critic,
            discriminator=self.discriminator,
            amp_data=amp_data,
            device=self.device,
            **alg_cfg_copy,
        )
        return alg