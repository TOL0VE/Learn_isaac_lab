# 引入你刚才贴的 skrl MotionLoader 代码
from .skrt_motion_loader import MotionLoader as SkrlMotionLoader 
import torch
import numpy as np

class AMPLoader:
    """
    这是一个适配器类。
    它内部使用 skrl 的 MotionLoader 来读取数据，
    对外提供 rsl_rl AMPOnPolicyRunner 所需的 feed_forward_generator 接口。
    """
    def __init__(
        self,
        device,
        dataset_path_root,  # 兼容 rsl_rl 参数，但在单文件模式下可能只用 datasets
        datasets,           # rsl_rl 传入的是 {"filename": weight}
        expected_joint_names, # rsl_rl 传入的机器人关节名称列表
        simulation_dt,      # 物理仿真步长
        slow_down_factor=1.0,
    ):
        self.device = device
        
        # 1. 处理文件路径 (skrl 只支持单文件，我们这里默认取 datasets 中的第一个文件)
        # 如果你想支持多文件混合，需要在这里实例化多个 SkrlMotionLoader 并加权采样
        if len(datasets) > 1:
            print(f"[Warning] SkrlMotionLoader 适配器目前仅支持单文件。将只使用: {list(datasets.keys())[0]}")
        
        filename = list(datasets.keys())[0]
        # 拼接完整路径
        import os
        motion_file = os.path.join(dataset_path_root, filename)
        
        # 2. 实例化 skrl 的 Loader
        self.skrl_loader = SkrlMotionLoader(motion_file=motion_file, device=device)
        
        # 3. 计算关节映射 (Re-indexing)
        # AMP 需要保证 expert 数据的关节顺序和仿真中机器人的关节顺序一致
        # skrl loader 提供了 get_dof_index 方法
        try:
            self.dof_indexes = self.skrl_loader.get_dof_index(expected_joint_names)
            # 对于 AMP，通常不需要 body index，除非你的判别器用了 keypoint 位置
            # 如果需要 body，同理：self.body_indexes = self.skrl_loader.get_body_index(...)
        except AssertionError as e:
            print(f"[Error] 动作文件中的关节名称与机器人不匹配！\n{e}")
            raise

        # 4. 计算采样参数
        # AMP 判别器通常需要 (s, s') 对，即当前帧和下一帧
        # skrl loader 采样是基于时间的，我们需要根据 dt 计算时间差
        self.dt = simulation_dt

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """
        实现 rsl_rl 需要的生成器接口。
        每次 yield 一个批次的专家数据 (Expert Batch)。
        """
        batch_size = num_mini_batch * mini_batch_size
        
        # 1. 随机采样时间点
        # 我们需要两组时间：t 和 t+dt
        times = self.skrl_loader.sample_times(batch_size)
        times_next = times + self.dt
        
        # 2. 从 skrl loader 获取数据 (Batch, ...)
        # sample 返回: (dof_pos, dof_vel, body_pos, body_rot, lin_vel, ang_vel)
        s_t = self.skrl_loader.sample(batch_size, times=times)
        s_next = self.skrl_loader.sample(batch_size, times=times_next)
        
        # 3. 提取并重组数据以构建 AMP 观测向量
        # 注意：这里的构建逻辑必须和你的 Environment._compute_amp_observations 严格一致！
        # 假设你的 AMP 观测包含：[DOF_Pos, DOF_Vel, Root_Height, Root_Rot, Root_Lin_Vel, Root_Ang_Vel, Key_Body_Pos]
        # 下面是一个通用的 Humanoid AMP 观测构建示例：
        
        expert_amp_obs_t = self._build_amp_obs(s_t)
        expert_amp_obs_next = self._build_amp_obs(s_next)
        
        # 4. 按 mini-batch 切分并 yield
        # rsl_rl 这里通常一次性把所有数据 yield 出去或者分批
        # 根据 Runner 逻辑，这里应该是一个生成器
        
        # 这里的实现方式取决于 Runner 是怎么调用的。
        # 如果 Runner 是： for batch in generator: ...
        # 我们需要把大数据切碎
        
        total_samples = expert_amp_obs_t.shape[0]
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        for i in range(num_mini_batch):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            idx = indices[start:end]
            
            # 产出 (s, s')
            # 注意：runner 那边接收的是 (expert_state, expert_next_state)
            yield expert_amp_obs_t[idx], expert_amp_obs_next[idx]

    def _build_amp_obs(self, sample_tuple):
        """
        将 skrl sample 返回的元组转换为 AMP 观测张量。
        这个函数极其重要，必须根据你的具体任务修改！
        """
        (dof_pos, dof_vel, body_pos, body_rot, lin_vel, ang_vel) = sample_tuple
        
        # [关键] 重排 DOF 顺序以匹配机器人
        dof_pos = dof_pos[:, self.dof_indexes]
        dof_vel = dof_vel[:, self.dof_indexes]
        
        # 假设 AMP 观测定义为：
        # [Root_Pos_Z (1), Root_Rot (4/6), Lin_Vel (3), Ang_Vel (3), DOF_Pos (N), DOF_Vel (N), Key_Body_Pos (M*3)]
        
        # 根节点通常是 body_names 里的第一个，或者需要通过 get_body_index 查找 "pelvis"/"torso"
        # skrl loader 的 body 数据是 (N, Num_Bodies, 3/4)
        root_idx = 0 # 假设根节点是第0个 body
        
        root_pos = body_pos[:, root_idx, :]
        root_rot = body_rot[:, root_idx, :] # Quaternion (w, x, y, z)
        root_lin_vel = lin_vel[:, root_idx, :]
        root_ang_vel = ang_vel[:, root_idx, :]
        
        # 注意：skrl 的四元数是 wxyz，IsaacGym/RSL-RL 很多时候也是 wxyz，但也可能是 xyzw
        # 需要确认环境中的转换逻辑。假设都是 wxyz。
        
        # 还需要转换 root_rot 到 tangent/normal 或者直接用 quat
        # 这里仅作示例拼接：
        # 排除 root_pos_x, root_pos_y (AMP 不关心绝对位置)
        amp_obs = torch.cat([
            root_pos[:, 2:3], # Z height
            root_rot,
            root_lin_vel,
            root_ang_vel,
            dof_pos,
            dof_vel
        ], dim=-1)
        
        return amp_obs