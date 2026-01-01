# MotionLoader 源码详解文档

## 1. 模块概述

**文件路径**: `motion_loader.py`
**功能**: 读取 `.npy` 格式的动作数据文件，将其转换为 PyTorch Tensor 存储在指定设备（CPU/GPU）上，并提供基于时间的**插值采样**功能。它是 AMP（Adversarial Motion Priors）算法中生成参考动作的关键组件。

---

## 2. 导入与类定义

```python
import numpy as np
import os
import torch
from typing import Optional

class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    帮助类：用于从 NumPy 文件格式加载和采样动作数据。
    """

```

* **依赖**:
* `numpy`: 用于读取原始 `.npy` 文件。
* `torch`: 用于将数据存储为张量，利用 GPU 加速计算（因为强化学习训练通常在 GPU 上进行）。
* `typing`: 用于类型提示，增加代码可读性。



---

## 3. 初始化 (`__init__`)

这是最关键的部分，负责数据搬运（Disk -> RAM -> VRAM）。

```python
    def __init__(self, motion_file: str, device: torch.device) -> None:
        """
        加载动作文件并初始化内部变量。
        """
        # 1. 校验文件是否存在
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        
        # 2. 使用 Numpy 读取数据 (CPU 内存)
        data = np.load(motion_file)

        self.device = device
        
        # 3. 解析元数据 (Metadata)
        # dof_names: 关节名称列表 (如 'left_knee', 'right_elbow')
        # body_names: 刚体名称列表 (如 'torso', 'head')
        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        # 4. 将物理数据转换为 Tensor 并移动到指定设备 (如 CUDA)
        # 这样做的好处是后续采样计算全部在 GPU 完成，避免 CPU-GPU 传输瓶颈
        
        # 关节位置 (Joint Positions): [Frames, Num_DOFs]
        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        # 关节速度 (Joint Velocities)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        # 身体位置 (Root/Body Positions): [Frames, Num_Bodies, 3]
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        # 身体旋转 (Rotations): [Frames, Num_Bodies, 4] (四元数格式 w,x,y,z)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        # 线速度与角速度
        self.body_linear_velocities = torch.tensor(
            data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        # 5. 计算时间相关属性
        # dt: 每一帧的时间间隔 (例如 60FPS -> dt=0.0166s)
        self.dt = 1.0 / data["fps"]
        self.num_frames = self.dof_positions.shape[0]
        # duration: 动作总时长
        self.duration = self.dt * (self.num_frames - 1)
        
        print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

```

---

## 4. 属性访问器 (Properties)

提供只读属性，方便外部获取骨架信息。

```python
    @property
    def dof_names(self) -> list[str]:
        """骨架 DOF 名称列表"""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """骨架刚体名称列表"""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """DOF 总数"""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """刚体总数"""
        return len(self._body_names)

```

---

## 5. 线性插值 (`_interpolate`)

用于处理**非旋转**数据（如位置、速度）的平滑过渡。

```python
    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        在两个连续值之间进行线性插值 (Linear Interpolation)。
        公式: result = (1 - blend) * a + blend * b
        """
        # 如果传入的是索引 (start/end)，则从 tensor a 中取出对应帧的数据
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        
        # 维度处理：确保 blend 系数能正确广播 (Broadcasting) 到数据维度
        # 例如数据是 [Batch, 3]，blend 是 [Batch]，需要变为 [Batch, 1]
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
            
        # 执行标准线性插值
        return (1.0 - blend) * a + blend * b

```

---

## 6. 球面线性插值 (`_slerp`)

**这是最复杂的数学部分**。用于处理**四元数旋转**。普通的线性插值会导致旋转变形（非单位四元数）或角速度不均匀，Slerp 保证了旋转是沿着球面最短路径均匀进行的。

```python
    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        四元数球面线性插值 (Spherical Linear Interpolation)。
        """
        # 1. 索引处理：取出这一帧(q0)和下一帧(q1)的四元数
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        
        # 2. 维度广播处理
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # 定义四元数分量顺序 w, x, y, z

        # 3. 计算点积 (Dot Product) = cos(theta)
        # 衡量两个旋转之间的夹角
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        # 4. 最短路径处理 (Flip)
        # 四元数 q 和 -q 表示相同的旋转。
        # 如果点积 < 0，说明夹角大于 90度，走长路径了。
        # 必须将 q1 取反，强制走最短路径插值。
        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        # 5. 计算角度 theta 和 sin(theta)
        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        # 6. 计算 Slerp 系数
        # 公式: q(t) = (sin((1-t)theta)/sin(theta)) * q0 + (sin(t*theta)/sin(theta)) * q1
        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        # 7. 组合结果
        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]
        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)

        # 8. 稳定性处理 (小角度近似)
        # 当角度非常小 (sin 接近 0) 时，除法会不稳定。
        # 此时退化为线性插值 (Lerp) 是安全的。
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        # 当两个旋转完全相同时，直接返回 q0
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        
        return new_q

```

---

## 7. 帧混合逻辑 (`_compute_frame_blend`)

计算给定时间点 `t` 落在第几帧和第几帧之间，以及偏移量。

```python
    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算给定时间对应的 帧索引(index) 和 混合系数(blend)。
        Args:
            times: 查询的时间数组 (秒)
        Returns:
            index_0: 左侧帧索引
            index_1: 右侧帧索引
            blend: 混合比例 [0, 1]
        """
        # 1. 归一化时间 Phase [0.0, 1.0]
        phase = np.clip(times / self.duration, 0.0, 1.0)
        
        # 2. 计算左侧帧索引 (index_0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        
        # 3. 计算右侧帧索引 (index_1)
        # 确保不越界，如果是最后一帧，右侧帧也是最后一帧
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        
        # 4. 计算混合系数 blend
        # blend = (当前时间 - 左侧帧时间) / 单帧间隔
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        
        return index_0, index_1, blend

```

---

## 8. 时间采样 (`sample_times`)

用于随机生成时间点，主要用于 `Reset` 时随机初始化机器人的状态。

```python
    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """
        在动作时长内均匀随机采样时间点。
        
        Args:
            num_samples: 需要采样多少个时间点 (通常等于 num_envs)
            duration: 指定最大采样时长 (默认是整个动作时长)
        """
        duration = self.duration if duration is None else duration
        # 安全检查
        assert (
            duration <= self.duration
        ), f"Specified duration ({duration}) > motion duration ({self.duration})"
        
        # 生成 [0, duration] 之间的随机数
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

```

---

## 9. 核心采样函数 (`sample`)

这是外部调用的主要接口。它整合了上述所有步骤：计算时间 -> 找帧 -> 插值 -> 返回状态。

```python
    def sample(
        self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> tuple[torch.Tensor, ...]:
        """
        采样动作数据。
        
        Returns:
            返回插值后的 关节位置, 关节速度, 身体位置, 身体旋转, 线速度, 角速度
        """
        # 1. 确定时间点
        # 如果没给 times，就随机生成
        times = self.sample_times(num_samples, duration) if times is None else times
        
        # 2. 计算混合参数
        index_0, index_1, blend = self._compute_frame_blend(times)
        
        # 将 numpy 的 blend 转为 tensor 以便在 GPU 计算
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        # 3. 执行插值并返回所有物理量
        # 除了旋转使用 _slerp，其他都使用 _interpolate
        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1), # 注意旋转
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        )

```

---

## 10. 辅助工具 (Indices Lookup)

将字符串名称转换为整数索引，因为仿真器底层只认索引。

```python
    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """根据关节名查找其在数据中的索引"""
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"DOF name ({name}) doesn't exist"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """根据刚体名查找其在数据中的索引"""
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"Body name ({name}) doesn't exist"
            indexes.append(self._body_names.index(name))
        return indexes

```

---

## 11. 测试入口 (`__main__`)

用于单独运行此脚本，检查动作文件是否能正常加载。

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    # 在 CPU 上加载测试
    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)

```