# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Enable forward references for type hints (allows using types before they are defined)
# 开启类型注解的后向兼容（允许在定义类型之前使用它）
from __future__ import annotations

# Import MISSING to indicate required fields that users must provide
# 导入 MISSING，用于标记用户必须提供的必填字段
from dataclasses import MISSING
# Import Literal for type checking restricted string values
# 导入 Literal，用于限制字符串变量只能取特定的值
from typing import Literal

# Import configclass decorator to register these classes as configuration objects
# 导入 configclass 装饰器，将这些类注册为配置对象（支持 Hydra 等配置管理工具）
from isaaclab.utils import configclass

# Import RND (Random Network Distillation) configuration
# 导入 RND（随机网络蒸馏，用于好奇心驱动探索）的配置
from .rnd_cfg import RslRlRndCfg
# Import Symmetry configuration
# 导入对称性约束配置
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Policy configurations #
#########################

@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""
    # PPO Actor-Critic 网络的配置类。

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""
    # 策略类的名称。默认为 "ActorCritic"（标准的多层感知机 MLP 结构）。
    # RSL-RL 会根据这个名字去工厂模式里找对应的 Python 类实例化。

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""
    # 策略的初始噪声标准差。
    # 决定了训练开始时的探索程度。值越大，动作随机性越大。通常设为 1.0 或 0.5。

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""
    # 噪声标准差的类型。默认为 "scalar"（标量）。可选值："scalar" 或 "log"。

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Default is False."""
    # 是否使用状态依赖的标准差。默认为 False。
    # 如果为 True，动作的方差会根据当前观测值变化（更加复杂）；如果为 False，方差是一个独立的参数。

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""
    # 是否对 Actor 网络的观测进行归一化。
    # 通常设为 True，有助于神经网络更快收敛。

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""
    # 是否对 Critic 网络的观测进行归一化。

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""
    # Actor 网络的隐藏层维度。
    # 例如 [512, 256, 128] 表示三层全连接层，节点数分别为 512, 256, 128。

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""
    # Critic 网络的隐藏层维度。通常与 Actor 相同或更大。

    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    # Actor 和 Critic 网络的激活函数。
    # 常见值："elu", "relu", "tanh"。RSL-RL 常用 "elu"。

@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""
    # 带有循环层（RNN）的 PPO Actor-Critic 网络配置类。

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""
    # 策略类名称。默认为 "ActorCriticRecurrent"。
    # 使用 LSTM 时必须用这个配置类。

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""
    # 要使用的 RNN 类型。可选 "lstm" 或 "gru"。

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""
    # RNN 层的隐藏维度（例如 512）。这就是“记忆容量”的大小。

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""
    # RNN 的层数（通常为 1）。

############################
# Algorithm configurations #
############################

@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""
    # PPO 算法的配置类。

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""
    # 算法类名称。默认为 "PPO"。

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""
    # 每次更新时的学习轮数（Epochs）。
    # 在收集完一批数据后，会对这批数据重复训练多少轮。通常是 5。

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""
    # 每次更新时的 Mini-batch 数量。
    # 将收集到的数据切分成多少个小批次进行梯度下降。通常是 4。

    learning_rate: float = MISSING
    """The learning rate for the policy."""
    # 策略的学习率。控制参数更新的步长。通常在 1e-3 到 1e-5 之间。

    schedule: str = MISSING
    """The learning rate schedule."""
    # 学习率调度策略。
    # 例如 "adaptive"（自适应，根据 KL 散度调整）或 "fixed"（固定）。

    gamma: float = MISSING
    """The discount factor."""
    # 折扣因子 (gamma)。
    # 决定了 Agent 看多远。0.99 意味着关注长期回报。

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""
    # GAE 的 Lambda 参数。用于平衡偏差和方差。通常为 0.95。

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""
    # 熵损失系数。
    # 鼓励 Agent 探索。如果策略过早收敛（不再尝试新动作），可以调大这个值。通常为 0.01。

    desired_kl: float = MISSING
    """The desired KL divergence."""
    # 期望的 KL 散度。
    # 如果使用自适应学习率，当实际 KL 超过这个值时，学习率会降低。通常为 0.01。

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    # 最大梯度范数。
    # 用于梯度裁剪（Gradient Clipping），防止梯度爆炸。通常为 1.0。

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""
    # 价值损失系数。平衡 Policy Loss 和 Value Loss 的权重。通常为 1.0。

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""
    # 是否使用截断的价值损失。
    # 类似于 PPO 的策略截断，防止 Value Function 更新太剧烈。通常为 True。

    clip_param: float = MISSING
    """The clipping parameter for the policy."""
    # 策略的截断参数 (epsilon)。
    # PPO 的核心参数，限制新旧策略的差异。通常为 0.2。

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False."""
    # 是否在每个 mini-batch 内归一化优势函数 (Advantage)。默认为 False。
    # 如果为 True，只在 mini-batch 内归一化；否则在整个采集的轨迹上归一化。

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Default is None, in which case RND is not used."""
    # RND 配置。默认为 None（不使用）。RND 用于稀疏奖励环境的探索。

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""
    # 对称性配置。默认为 None（不使用）。用于强制策略保持物理对称性。
#########################
# Runner configurations #
#########################

@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""
    # Runner（运行器）的基础配置类。

    seed: int = 42
    """The seed for the experiment. Default is 42."""
    # 实验的随机种子。默认为 42。确保实验可复现。

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""
    # RL Agent 运行的设备。默认为 "cuda:0"（第一块 GPU）。

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""
    # 每次更新前，每个环境运行的步数。
    # 决定了 Rollout Buffer 的长度。例如 24。

    max_iterations: int = MISSING
    """The maximum number of iterations."""
    # 最大迭代次数。训练多少次 update 后停止。

    empirical_normalization: bool | None = None
    """This parameter is deprecated and will be removed in the future."""
    # 该参数已弃用，未来将被移除。
    # 请改用 `actor_obs_normalization` 和 `critic_obs_normalization`。

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets."""
    # 观测组到观测集合的映射。
    # 这是一个高级配置，用于将环境输出的字典（例如 "image", "proprio"）映射到 RSL-RL 需要的 "policy" 和 "critic" 输入。
    # 例如：policy 网络可能只需要关节数据，而 critic 网络可能需要特权信息（privileged info）。

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None."""
    # 动作截断值。如果为 None，则不进行截断。默认为 None。
    # 注意：这个截断是在 RslRlVecEnvWrapper 中执行的。

    save_interval: int = MISSING
    """The number of iterations between saves."""
    # 保存模型的间隔迭代次数。例如每 50 次保存一次。

    experiment_name: str = MISSING
    """The experiment name."""
    # 实验名称。用于生成日志文件夹路径。

    run_name: str = ""
    """The run name. Default is empty string."""
    # 运行名称。默认为空字符串。
    # 实际的日志目录名为 `{时间戳}_{run_name}`。

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""
    # 使用的日志记录器。默认为 "tensorboard"。
    # 也支持 "wandb" (Weights & Biases) 和 "neptune"。

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""
    # Neptune 项目名称。

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""
    # WandB 项目名称。

    resume: bool = False
    """Whether to resume a previous training. Default is False."""
    # 是否恢复之前的训练。默认为 False。

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all)."""
    # 要加载的运行目录。支持正则匹配。默认为 ".*"（加载最新的）。

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is "model_.*.pt" (all)."""
    # 要加载的检查点文件。支持正则匹配。默认为最新保存的模型。


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""
    # On-Policy 算法（如 PPO）的 Runner 配置类。

    class_name: str = "OnPolicyRunner"
    """The runner class name. Default is OnPolicyRunner."""
    # Runner 类名称。默认为 "OnPolicyRunner"。

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""
    # 策略配置（嵌套上面的 RslRlPpoActorCriticCfg 或其 Recurrent 版本）。

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
    # 算法配置（嵌套上面的 RslRlPpoAlgorithmCfg）。



###############################
# AMP Component configurations#
###############################

@configclass
class RslRlAmpDiscriminatorCfg:
    """Configuration for the AMP Discriminator network."""
    # AMP 判别器网络的配置类。

    class_name: str = "AMPOnPolicyRunner"
    """The discriminator class name."""
    # 判别器类名。

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the discriminator network."""
    # 判别器网络的隐藏层维度。
    # 通常比 Policy 网络大，例如 [1024, 512]。

    reward_scale: float = MISSING
    """The scale of the style reward."""
    # 风格奖励的缩放系数。
    # 控制 AMP 奖励对总奖励的贡献程度，非常关键的超参数。通常在 1.0 到 2.0 之间。

    loss_type: Literal["BCEWithLogits", "Wasserstein", "LeastSquares"] = "BCEWithLogits"
    """The loss function used for the discriminator."""
    # 判别器使用的损失函数类型。默认为 BCEWithLogits。

    empirical_normalization: bool = False
    """Whether to normalize the AMP inputs empirically."""
    # 是否对 AMP 的输入（观察值）进行经验归一化。


@configclass
class RslRlAmpLoaderCfg:
    """Configuration for the AMP Expert Data Loader."""
    # AMP 专家数据加载器的配置类。

    amp_data_path: str = MISSING
    """Root path to the directory containing expert motion files (.npy)."""
    # 包含专家动作文件 (.npy) 的根目录路径。

    datasets: dict[str, float] = MISSING
    """A dictionary mapping dataset filenames to their sampling weights."""
    # 专家数据集映射表。Key 是文件名，Value 是采样权重。
    # 例如: {"walk.npy": 1.0, "run.npy": 0.5}

    slow_down_factor: float = 1.0
    """Factor to slow down the reference motion playback."""
    # 动作放慢倍率。默认为 1.0 (原速)。


##################################
# AMP Algorithm configurations   #
##################################

@configclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP-PPO algorithm (Inherits from PPO)."""
    # AMP-PPO 算法配置类（继承自标准 PPO 配置）。

    #?
    class_name: str = "AMP_PPO"
    """The algorithm class name."""
    # 算法类名，必须与 AMPOnPolicyRunner 中实例化的类名一致。

    amp_replay_buffer_size: int = 100000
    """Size of the replay buffer for policy-generated motions."""
    # 用于存储策略生成动作的 Replay Buffer 大小。
    # 判别器训练需要对比 Buffer 中的假数据和 Loader 中的真数据。

    min_normalized_std: float = 1e-6
    """Minimum standard deviation for normalization to avoid division by zero."""
    # 归一化时的最小标准差，防止除以零。


#############################
# AMP Runner configurations #
#############################

@configclass
class RslRlAmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for AMP on-policy algorithms."""
    # AMP On-Policy 算法的 Runner 配置类。

    class_name: str = "AMPOnPolicyRunner"
    """The runner class name."""
    # Runner 类名。

    # Override the algorithm config to use AMP-PPO
    algorithm: RslRlAmpPpoAlgorithmCfg = MISSING
    """The AMP algorithm configuration."""
    # 算法配置，强制指定为 RslRlAmpPpoAlgorithmCfg。

    # Add Discriminator config
    discriminator: RslRlAmpDiscriminatorCfg = MISSING
    """The discriminator configuration."""
    # 判别器配置。

    # Add Dataset/Loader config
    dataset: RslRlAmpLoaderCfg = MISSING
    """The expert dataset configuration."""
    # 专家数据集配置。