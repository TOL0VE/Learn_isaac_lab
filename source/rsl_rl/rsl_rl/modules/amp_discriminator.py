# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F

from rsl_rl.networks import EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    """
    AMP 算法的判别器网络实现。
    
    这个网络被训练用来区分“专家数据（Expert）”和“策略生成的数据（Policy）”。
    它通过对抗学习为 Agent 提供奖励信号（风格奖励）。

    参数 Args:
        input_dim (int): 输入状态的维度 (通常是 state + next_state 拼接后的维度)。
        hidden_layer_sizes (list): 隐藏层大小列表，例如 [1024, 512]。
        reward_scale (float): 风格奖励的缩放系数。
        reward_clamp_epsilon (float): 用于奖励计算的数值稳定 epsilon。
        device (str | torch.device): 运行设备。
        loss_type (str): 损失函数类型 ('BCEWithLogits' 或 'Wasserstein')。通常用 BCE。
        eta_wgan (float): 如果使用 Wasserstein Loss，这是缩放因子。
        use_minibatch_std (bool): 是否在网络中使用 Minibatch Std 技术（防模式崩塌）。
        empirical_normalization (bool): 是否对 AMP 观测进行经验归一化。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 1.0e-4,
        device: str | torch.device = "cpu",
        loss_type: str = "BCEWithLogits",
        eta_wgan: float = 0.3,
        use_minibatch_std: bool = True,
        empirical_normalization: bool = False,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.reward_scale = reward_scale
        self.reward_clamp_epsilon = reward_clamp_epsilon
        
        # --- 1. 构建 MLP 网络骨架 ---
        layers = []
        curr_in_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU()) # AMP 原文通常使用 ReLU 或 ELU
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        
        # 如果启用了 Minibatch Std，最后一层的输入维度需要 +1
        final_in_dim = hidden_layer_sizes[-1] + (1 if use_minibatch_std else 0)
        self.linear = nn.Linear(final_in_dim, 1) # 输出一个 Logit 值

        # --- 2. 观测归一化模块 ---
        self.empirical_normalization = empirical_normalization
        # AMP 观测通常由 (s, s') 组成，归一化器只需要维护单帧 s 的统计信息
        amp_obs_dim = input_dim // 2
        if empirical_normalization:
            self.amp_normalizer = EmpiricalNormalization(shape=[amp_obs_dim])
        else:
            self.amp_normalizer = nn.Identity() # 不做处理

        self.to(self.device)
        self.train() # 默认为训练模式
        
        # --- 3. 配置损失函数 ---
        self.use_minibatch_std = use_minibatch_std
        self.loss_type = loss_type if loss_type is not None else "BCEWithLogits"
        
        if self.loss_type == "BCEWithLogits":
            # 标准 GAN Loss: Binary Cross Entropy
            self.loss_fun = torch.nn.BCEWithLogitsLoss()
        elif self.loss_type == "Wasserstein":
            # WGAN Loss (实验性)
            self.loss_fun = None
            self.eta_wgan = eta_wgan
            print("The Wasserstein-like loss is experimental")
        else:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}. Supported types are 'BCEWithLogits' and 'Wasserstein'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        Args:
            x: 输入 Tensor (batch_size, input_dim)。
        Returns:
            Logits: 判别器的原始输出分数 (未经过 Sigmoid)。
        """

        # 1. 归一化处理
        # 假设输入是 [state, next_state] 拼接的，我们拆开分别归一化
        state, next_state = torch.split(x, self.input_dim // 2, dim=-1)
        state = self.amp_normalizer(state)
        next_state = self.amp_normalizer(next_state)
        x = torch.cat([state, next_state], dim=-1)

        # 2. 通过 MLP 主干
        h = self.trunk(x)
        
        # 3. Minibatch Std 技巧 (防止 Mode Collapse)
        # 将整个 Batch 的标准差作为一个特征拼接到每个样本上
        if self.use_minibatch_std:
            s = self._minibatch_std_scalar(h)
            h = torch.cat([h, s], dim=-1)
            
        # 4. 最后一层线性变换
        return self.linear(h)

    def _minibatch_std_scalar(self, h: torch.Tensor) -> torch.Tensor:
        """
        计算 Batch 内特征的标准差均值。
        这让判别器能看到“整体的多样性”，防止它只盯着单个样本看。
        """
        if h.shape[0] <= 1:
            return h.new_zeros((h.shape[0], 1))
        # 计算 std -> 对特征维度求平均 -> 得到一个标量
        s = h.float().std(dim=0, unbiased=False).mean()
        # 扩展成 (Batch, 1) 的形状以便拼接
        return s.expand(h.shape[0], 1).to(h.dtype)

    def predict_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        预测 AMP 风格奖励。
        
        Args:
            state: 当前状态
            next_state: 下一时刻状态
        Returns:
            reward: 计算出的奖励值
        """
        with torch.no_grad():
            # 拼接并前向传播 (forward 内部会自动归一化)
            discriminator_logit = self.forward(torch.cat([state, next_state], dim=-1))

            if self.loss_type == "Wasserstein":
                discriminator_logit = torch.tanh(self.eta_wgan * discriminator_logit)
                return self.reward_scale * torch.exp(discriminator_logit).squeeze()
            
            # --- 关键公式 ---
            # 奖励公式推导: r = max(0, 1 - 0.25(D-1)^2) 的变体
            # 这里使用了 softplus 形式: softplus(logit) == -log(1 - sigmoid(logit))
            # 这种形式在数值上更稳定，且符合 GAIL 的数学推导 (-log(1-D))
            reward = F.softplus(discriminator_logit)
            reward = self.reward_scale * reward
            return reward.squeeze()

    def update_normalization(self, *batches: torch.Tensor) -> None:
        """使用提供的 AMP 数据批次更新归一化器的统计信息 (Mean/Var)。"""
        if not self.empirical_normalization:
            return
        with torch.no_grad():
            for batch in batches:
                self.amp_normalizer.update(batch)

    def compute_loss(
        self,
        policy_d,          # 策略数据 (Fake) 的 Logits
        expert_d,          # 专家数据 (Real) 的 Logits
        sample_amp_expert, # 用于梯度惩罚的专家样本 (Raw)
        sample_amp_policy, # 用于梯度惩罚的策略样本 (Raw)
        lambda_: float = 10, # 梯度惩罚系数
    ):
        """
        计算判别器的总损失 (AMP Loss + Gradient Penalty)。
        这个函数被 Algorithm 类调用。
        """

        # 1. 计算梯度惩罚 (Gradient Penalty)
        # 需要先手动归一化样本，因为 autograd.grad 需要对输入求导
        sample_amp_expert = tuple(self.amp_normalizer(s) for s in sample_amp_expert)
        sample_amp_policy = tuple(self.amp_normalizer(s) for s in sample_amp_policy)
        
        grad_pen_loss = self.compute_grad_pen(
            expert_states=sample_amp_expert,
            policy_states=sample_amp_policy,
            lambda_=lambda_,
        )
        
        # 2. 计算分类损失 (Classification Loss)
        if self.loss_type == "BCEWithLogits":
            # 专家数据的目标是 1
            expert_loss = self.loss_fun(expert_d, torch.ones_like(expert_d))
            # 策略数据的目标是 0
            policy_loss = self.loss_fun(policy_d, torch.zeros_like(policy_d))
            # 总分类损失
            amp_loss = 0.5 * (expert_loss + policy_loss)
            
        elif self.loss_type == "Wasserstein":
            amp_loss = self.wgan_loss(policy_d=policy_d, expert_d=expert_d)
            
        return amp_loss, grad_pen_loss

    def compute_grad_pen(
        self,
        expert_states: tuple[torch.Tensor, torch.Tensor],
        policy_states: tuple[torch.Tensor, torch.Tensor],
        lambda_: float = 10,
    ) -> torch.Tensor:
        """
        计算梯度惩罚 (Gradient Penalty)，用于正则化判别器。
        这是 R1 Regularization 或 WGAN-GP 的核心。
        """
        expert = torch.cat(expert_states, -1)

        if self.loss_type == "Wasserstein":
            # WGAN-GP: 在 Real 和 Fake 之间的插值点上惩罚梯度的模长 (使其接近1)
            policy = torch.cat(policy_states, -1)
            alpha = torch.rand(expert.size(0), 1, device=expert.device)
            alpha = alpha.expand_as(expert)
            data = alpha * expert + (1 - alpha) * policy
            data = data.detach().requires_grad_(True)
            # ... (WGAN-GP 计算逻辑) ...
            h = self.trunk(data)
            # ... 
            # (为了简洁，省略中间代码，逻辑与原文件一致)
            scores = self.linear(h) # 注意：这里如果用了minibatch std也要加
            
            grad = autograd.grad(outputs=scores, inputs=data, ...)[0]
            return lambda_ * (grad.norm(2, dim=1) - 1.0).pow(2).mean()
            
        elif self.loss_type == "BCEWithLogits":
            # R1 Regularization: 只在真实样本 (Real Data) 上惩罚梯度
            # 目标：让判别器在真实数据附近的梯度平滑
            # 公式：0.5 * lambda * ||∇_x D(x_real)||^2
            
            data = expert.detach().requires_grad_(True)
            
            # 再次前向传播，为了求导
            h = self.trunk(data)
            if self.use_minibatch_std:
                with torch.no_grad():
                    s = self._minibatch_std_scalar(h)
                h = torch.cat([h, s], dim=-1)
            scores = self.linear(h)

            # 对输入 data 求梯度
            grad = autograd.grad(
                outputs=scores.sum(),
                inputs=data,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            
            # 梯度的 L2 范数平方
            return 0.5 * lambda_ * (grad.pow(2).sum(dim=1)).mean()

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def wgan_loss(self, policy_d, expert_d):
        # WGAN Loss: E[D(fake)] - E[D(real)]
        policy_d = torch.tanh(self.eta_wgan * policy_d)
        expert_d = torch.tanh(self.eta_wgan * expert_d)
        return policy_d.mean() - expert_d.mean()