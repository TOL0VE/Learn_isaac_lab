# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticCNN
from rsl_rl.storage import RolloutStorage

# 导入 AMP 相关的组件 (确保这些路径在你的项目中存在)
from rsl_rl_amp.storage import ReplayBuffer
from rsl_rl_amp.networks import Discriminator
from rsl_rl_amp.utils import AMPLoader


class AMP_PPO:
    """
    AMP_PPO 实现了对抗运动先验 (AMP) 与 PPO 的结合。
    
    该类参考了标准 PPO 的实现，增加了对 AMP 判别器的训练和风格奖励的计算。
    同时支持 MLP、CNN 和 Recurrent (LSTM/GRU) 策略网络。
    """


    actor_critic: Union[ActorCritic, ActorCriticRecurrent, ActorCriticCNN]

    def __init__(
        self,
        actor_critic: Union[ActorCritic, ActorCriticRecurrent, ActorCriticCNN],
        discriminator: Discriminator,
        amp_data: AMPLoader,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        amp_replay_buffer_size: int = 100000,
        use_smooth_ratio_clipping: bool = False,
        device: str = "cpu",
    ) -> None:
        # 设置设备和基本超参数
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # 初始化判别器并移动到设备
        self.discriminator = discriminator.to(self.device)
        
        # 初始化 AMP 专用的 Transition (用于暂存 AMP 观测)
        self.amp_transition = RolloutStorage.Transition()
        
        # 确定 Replay Buffer 的观测维度
        # 判别器输入通常是 (s, s') 拼接，Buffer 只存 s，所以除以 2
        obs_dim = self.discriminator.input_dim // 2
        self.amp_storage = ReplayBuffer(
            obs_dim=obs_dim, buffer_size=amp_replay_buffer_size, device=device
        )
        self.amp_data = amp_data

        # 初始化策略网络 (Actor-Critic)
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        
        # Storage 将在 init_storage 中初始化
        self.storage: Optional[RolloutStorage] = None

        # 创建优化器
        # 注意：这里我们为判别器的不同部分设置了不同的权重衰减 (Weight Decay)
        params = [
            {"params": self.actor_critic.parameters(), "name": "actor_critic"},
            {
                "params": self.discriminator.trunk.parameters(),
                "weight_decay": 10e-4,
                "name": "amp_trunk",
            },
            {
                "params": self.discriminator.linear.parameters(),
                "weight_decay": 10e-2,
                "name": "amp_head",
            },
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        # 初始化 PPO 转移对象
        self.transition = RolloutStorage.Transition()
        
        # PPO 算法参数
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_smooth_ratio_clipping = use_smooth_ratio_clipping

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        observations: TensorDict,
        action_shape: Tuple[int, ...],
    ) -> None:
        """初始化 Rollout Storage。"""
        self.storage = RolloutStorage(
            training_type="rl",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=observations,
            actions_shape=action_shape,
            device=self.device,
        )

    def test_mode(self) -> None:
        """切换到评估模式。"""
        self.actor_critic.eval()

    def train_mode(self) -> None:
        """切换到训练模式。"""
        self.actor_critic.train()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """执行动作选择。"""
        # [修改 3] 处理递归网络 (RNN/LSTM) 的隐藏状态
        if self.actor_critic.is_recurrent:
            # 获取当前隐藏状态并存入 transition，供 storage 记录
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        # 计算动作和价值 (detach 切断梯度，因为这是数据收集阶段)
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        
        return self.transition.actions

    def act_amp(self, amp_obs: torch.Tensor) -> None:
        """存储 AMP 观测 (用于后续插入 Replay Buffer)。"""
        self.amp_transition.observations = amp_obs

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: Dict[str, Any],
    ) -> None:
        """处理环境返回的步骤信息。"""
        # 更新观测归一化统计量
        self.actor_critic.update_normalization(obs)

        # 记录奖励和结束标志
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # 处理超时 (Time Outs) 的 Bootstrapping
        # 如果是超时结束，需要把下一时刻的价值加回去
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # 添加到 Storage
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        
        # [修改 4] 重置结束环境的 RNN 状态
        # actor_critic.reset 会处理 LSTM hidden states 的清零
        self.actor_critic.reset(dones)

    def process_amp_step(self, amp_obs: torch.Tensor) -> None:
        """将 AMP 观测插入 Replay Buffer。"""
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.amp_transition.clear()

    def compute_returns(self, obs: TensorDict) -> None:
        """计算 GAE 回报。"""
        last_values = self.actor_critic.evaluate(obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(
        self,
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """执行一次完整的更新 (PPO + AMP)。"""
        
        # 初始化统计变量
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_accuracy_policy = 0.0
        mean_accuracy_expert = 0.0
        mean_accuracy_policy_elem = 0.0
        mean_accuracy_expert_elem = 0.0
        mean_kl_divergence = 0.0

        # [修改 5] 根据策略类型选择 Mini-batch 生成器
        if self.actor_critic.is_recurrent:
            # 如果是递归网络，使用 recurrent 生成器 (处理序列数据)
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            # 否则使用标准生成器
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # AMP 策略数据生成器
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            num_mini_batch=self.num_learning_epochs * self.num_mini_batches,
            mini_batch_size=self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
            allow_replacement=True,
        )

        # AMP 专家数据生成器
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )

        # 开始循环更新
        for sample, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            # 解包 PPO 样本数据
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch, # 这里包含了 hidden states
                masks_batch,
            ) = sample

            # [修改 6] 按照你提供的 PPO 代码逻辑，重新计算当前策略的输出
            # 如果是递归网络，需要传入 hidden_state
            # 注意：PPO 代码中 hidden_states_batch[0] 是 Actor 的状态，[1] 是 Critic 的状态
            if self.actor_critic.is_recurrent:
                self.actor_critic.act(
                    obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0]
                )
            else:
                self.actor_critic.act(obs_batch, masks=masks_batch)
                
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            
            # 重新计算 Value
            if self.actor_critic.is_recurrent:
                value_batch = self.actor_critic.evaluate(
                    obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
                )
            else:
                value_batch = self.actor_critic.evaluate(
                    obs_batch, masks=masks_batch
                )
                
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # --- 自适应学习率 (KL Divergence) ---
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    mean_kl_divergence += kl_mean.item()

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # --- 计算 PPO Surrogate Loss ---
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )

            # 裁剪比率
            min_ = 1.0 - self.clip_param
            max_ = 1.0 + self.clip_param
            
            if self.use_smooth_ratio_clipping:
                clipped_ratio = (
                    1
                    / (1 + torch.exp((-(ratio - min_) / (max_ - min_) + 0.5) * 4))
                    * (max_ - min_)
                    + min_
                )
            else:
                clipped_ratio = torch.clamp(ratio, min_, max_)

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * clipped_ratio
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # --- 计算 Value Loss ---
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # PPO 总损失
            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # --- 处理 AMP 判别器损失 ---
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            # 确保在正确设备上
            policy_state = policy_state.to(self.device)
            policy_next_state = policy_next_state.to(self.device)
            expert_state = expert_state.to(self.device)
            expert_next_state = expert_next_state.to(self.device)

            # 保存原始数据用于归一化更新 (detach)
            policy_state_raw = policy_state.detach().clone()
            policy_next_state_raw = policy_next_state.detach().clone()
            expert_state_raw = expert_state.detach().clone()
            expert_next_state_raw = expert_next_state.detach().clone()

            # 拼接输入给判别器 (s, s')
            B_pol = policy_state.size(0)
            discriminator_input = torch.cat(
                (
                    torch.cat([policy_state, policy_next_state], dim=-1),
                    torch.cat([expert_state, expert_next_state], dim=-1),
                ),
                dim=0,
            )
            
            discriminator_output = self.discriminator(discriminator_input)
            policy_d, expert_d = (
                discriminator_output[:B_pol],
                discriminator_output[B_pol:],
            )

            # 计算 AMP 损失 (分类 + 梯度惩罚)
            amp_loss, grad_pen_loss = self.discriminator.compute_loss(
                policy_d=policy_d,
                expert_d=expert_d,
                sample_amp_expert=(expert_state, expert_next_state),
                sample_amp_policy=(policy_state, policy_next_state),
                lambda_=10,
            )

            # 最终联合损失
            loss = ppo_loss + (amp_loss + grad_pen_loss)

            # --- 反向传播 ---
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # --- 更新 AMP 归一化器 ---
            self.discriminator.update_normalization(
                expert_state_raw,
                expert_next_state_raw,
                policy_state_raw,
                policy_next_state_raw,
            )

            # --- 统计数据 ---
            policy_d_prob = torch.sigmoid(policy_d)
            expert_d_prob = torch.sigmoid(expert_d)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d_prob.mean().item()
            mean_expert_pred += expert_d_prob.mean().item()

            mean_accuracy_policy += torch.sum(
                torch.round(policy_d_prob) == torch.zeros_like(policy_d_prob)
            ).item()
            mean_accuracy_expert += torch.sum(
                torch.round(expert_d_prob) == torch.ones_like(expert_d_prob)
            ).item()

            mean_accuracy_expert_elem += expert_d_prob.numel()
            mean_accuracy_policy_elem += policy_d_prob.numel()

        # 计算平均值
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_accuracy_policy /= max(1, mean_accuracy_policy_elem)
        mean_accuracy_expert /= max(1, mean_accuracy_expert_elem)
        mean_kl_divergence /= num_updates

        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
            mean_accuracy_policy,
            mean_accuracy_expert,
            mean_kl_divergence,
        )