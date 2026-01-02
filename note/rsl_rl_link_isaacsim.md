# 1 obs

```python
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
```

wrap operation

```python
    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment."""
        # 仅获取观测而不 step（通常用于初始化或调试）
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])
```


```python
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
```

# 2 action

```python
obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
```


>source/rsl_rl/rsl_rl/algorithms/ppo.py

```python
    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions
```

>source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py

```python
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
```

