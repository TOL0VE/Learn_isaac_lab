import torch
import numpy as np

class AMPLoader:
    def __init__(
        self,
        env,  # 传入环境实例
        device,
        time_between_frames, # dt
    ):
        self.env = env
        self.device = device
        self.dt = time_between_frames

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """
        生成器：调用 Env 的方法来获取 (Expert_State, Expert_Next_State)
        """
        batch_size = num_mini_batch * mini_batch_size
        
        # 1. 采样随机时间点 (Batch)
        # 我们需要访问 env 内部的 motion_loader 来采样时间
        # 假设 env.unwrapped 有 _motion_loader 属性
        motion_loader = self.env.unwrapped._motion_loader
        times = motion_loader.sample_times(batch_size)
        
        # 计算下一帧的时间 (用于 Next State)
        times_next = times + self.dt

        # 2. 调用 Env 的方法获取数据
        # 注意：collect_reference_motions 应该处理好了历史帧堆叠和 compute_obs
        
        # 获取 s_t
        expert_amp_obs = self.env.unwrapped.collect_reference_motions(
            num_samples=batch_size, 
            current_times=times
        )
        
        # 获取 s_{t+1}
        expert_amp_obs_next = self.env.unwrapped.collect_reference_motions(
            num_samples=batch_size, 
            current_times=times_next
        )

        # 确保数据在正确的设备上
        expert_amp_obs = expert_amp_obs.to(self.device)
        expert_amp_obs_next = expert_amp_obs_next.to(self.device)

        # 3. 打乱索引 (Shuffle)
        indices = torch.randperm(batch_size, device=self.device)
        
        # 4. 分批 Yield
        for i in range(num_mini_batch):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            batch_idx = indices[start:end]
            
            yield expert_amp_obs[batch_idx], expert_amp_obs_next[batch_idx]