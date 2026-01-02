# Copyright (c) 2022-2025, The Isaac Lab Project Developers ...
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

# 从 RSL-RL 库中导入 VecEnv 抽象基类。
# 所有的环境如果想被 RSL-RL 的 PPO Runner 运行，都必须继承这个类。
from rsl_rl.env import VecEnv

# 导入 Isaac Lab 的两种环境基类：
# DirectRLEnv: 直接物理环境（如你现在用的 Humanoid），逻辑全在一个文件里。
# ManagerBasedRLEnv: 基于管理器的复杂环境。
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class RslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for the RSL-RL library
    用于将 Isaac Lab 环境包装成 RSL-RL 库可用的格式。

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.
        小心：这个包装器必须是链条中的最后一个。因为它继承自 RSL-RL 的 VecEnv 而不是标准的 Gym Wrapper。
        如果后面还有其他 Wrapper，可能无法识别它。

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        """Initializes the wrapper.
        初始化包装器。

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.
            注意：包装器会在初始化时主动调用一次 reset，因为 RSL-RL 的运行器默认假设环境已经是 ready 状态。

        Args:
            env: The environment to wrap around. (要包装的环境实例)
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.
                          (动作裁剪值。例如设为 1.0，则网络输出会被裁到 [-1, 1]。)

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """

        # check that input is valid
        # 检查传入的 env 是否是 Isaac Lab 的标准环境类
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        # 获取环境数量 (num_envs)，并行训练的关键参数
        self.num_envs = self.unwrapped.num_envs
        # 获取设备 (cuda:0 或 cpu)
        self.device = self.unwrapped.device
        # 获取最大回合长度
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        # 获取动作空间维度 (num_actions)
        if hasattr(self.unwrapped, "action_manager"):
            # 如果是 Manager-Based 环境，从管理器获取维度
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            # 如果是 Direct 环境，直接展平动作空间计算维度
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        # 修改环境的动作空间定义（详见底部的 _modify_action_space 函数）
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        # 立即重置环境
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        # 获取环境配置对象
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.
        返回剥离了所有包装器的最底层环境对象。
        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        # 获取当前步数计数器
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.
        设置步数计数器。

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
            这是 RSL-RL 的一个特性：在训练开始时随机化每个环境的当前步数，防止所有环境同步 Reset 导致数据分布震荡。
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP (马尔可夫决策过程)
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        # reset the environment
        # 调用底层环境 reset，获取 obs 字典和 extras
        obs_dict, extras = self.env.reset()
        # [关键] 将 obs_dict 包装成 TensorDict 返回。RSL-RL 需要这种格式。
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        # 仅获取观测而不 step（通常用于初始化或调试）
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        # 如果设置了动作裁剪，在这里将动作 clamp 到范围内
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        
        # record step information
        # 执行环境的一步。
        # terminated: 失败（摔倒）
        # truncated: 超时（达到最大步数）
        # extras: 额外信息（你的 AMP 观测就在这里面的 "amp_obs" key 里！）
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        
        # compute dones for compatibility with RSL-RL
        # RSL-RL 不区分 terminated 和 truncated，统一用 dones 表示结束。
        # 用逻辑或 (|) 合并，并转为 long 类型。
        dones = (terminated | truncated).to(dtype=torch.long)
        
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        # 将超时信息存入 extras，这对计算 PPO 的价值估计（处理 Time Limit 问题）很重要。
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
            
        # return the step information
        # 返回 RSL-RL 兼容的四元组
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        # 如果启用了 clip_actions，修改 Gym 的 action_space 定义以匹配新的范围。
        # 这主要用于那些依赖 action_space.high/low 来进行初始化的算法。
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other
        #   action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )