# MotionViewer 源码详解文档

## 1. 模块概述

**文件功能**: `motion_viewer.py` 是一个轻量级的 3D 动作可视化工具。
**核心作用**: 读取通过 `MotionLoader` 加载的骨架位置数据，利用 `matplotlib.animation` 制作 3D 散点图动画，直观展示机器人的动作（如行走、跑步、跳舞）。
**使用场景**:

* 检查 `.npy` 动作文件是否损坏。
* 验证动作的坐标轴方向是否正确（Z轴向上还是Y轴向上）。
* 观察动作的范围和幅度。

## 2. 依赖库

* **matplotlib**: 核心绘图库，用于 3D 绘图和动画。
* **numpy**: 处理数值计算。
* **torch**: 因为 `MotionLoader` 返回的是 Tensor，需要它来进行数据转换。
* **MotionLoader**: 自定义模块，负责读取底层数据文件。

---

## 3. 详细代码注释与解析

以下按照代码执行顺序进行详细解读：

### 3.1 导入与兼容性处理

```python
from __future__ import annotations

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch

# 导入 3D 绘图工具包 (必须导入，否则 projection='3d' 会报错)
import mpl_toolkits.mplot3d  # noqa: F401

# 尝试导入同目录下的 MotionLoader
# try-except 块是为了兼容不同的包导入路径（直接运行脚本 vs作为模块导入）
try:
    from .motion_loader import MotionLoader
except ImportError:
    from motion_loader import MotionLoader

```

### 3.2 MotionViewer 类定义

```python
class MotionViewer:
    """
    辅助类：用于可视化 NumPy 文件格式的动作数据。
    """

    def __init__(self, motion_file: str, device: torch.device | str = "cpu", render_scene: bool = False) -> None:
        """
        加载动作文件并初始化内部变量。

        Args:
            motion_file: 动作文件的路径 (.npy)。
            device: 加载数据的设备 (通常 'cpu' 即可，因为绘图是在 CPU 上进行的)。
            render_scene: 渲染模式开关。
                - True (Scene View): 视角固定，包含整个动作的空间范围（适合观察移动轨迹）。
                - False (Skeleton View): 视角跟随骨架，聚焦于骨架本身的姿态（适合观察原地动作细节）。

        Raises:
            AssertionError: 如果指定的文件不存在。
        """
        self._figure = None
        self._figure_axes = None
        self._render_scene = render_scene

        # 1. 调用 MotionLoader 加载数据
        # 这会读取文件并计算出每一帧的关节位置、旋转等
        self._motion_loader = MotionLoader(motion_file=motion_file, device=device)

        # 2. 获取元数据
        self._num_frames = self._motion_loader.num_frames # 总帧数
        self._current_frame = 0 # 当前播放帧索引
        
        # 3. 数据转换 (GPU/Tensor -> CPU/Numpy)
        # Matplotlib 只能处理 Numpy 数组，不能直接处理 PyTorch Tensor
        self._body_positions = self._motion_loader.body_positions.cpu().numpy()

        # 4. 打印身体部件统计信息
        # 遍历所有身体部件（如手、脚、头），打印它们在动作过程中的 XYZ 最小和最大坐标
        # 这有助于快速检查数据是否有异常值（例如某个点飞到无穷远）
        print("\nBody")
        for i, name in enumerate(self._motion_loader.body_names):
            minimum = np.min(self._body_positions[:, i], axis=0).round(decimals=2)
            maximum = np.max(self._body_positions[:, i], axis=0).round(decimals=2)
            print(f"  |-- [{name}] minimum position: {minimum}, maximum position: {maximum}")

```

### 3.3 绘图回调函数 (`_drawing_callback`)

这是动画的核心，每一帧都会被 `FuncAnimation` 调用一次。

```python
    def _drawing_callback(self, frame: int) -> None:
        """
        每一帧调用的绘图回调函数。
        """
        # 1. 获取当前帧的骨架数据
        # vertices shape: [Num_Bodies, 3] -> (x, y, z) 坐标
        vertices = self._body_positions[self._current_frame]

        # 2. 清除上一帧的画面
        # 如果不清除，新的点会叠加在旧的点上，导致画面变成一团黑
        self._figure_axes.clear()

        # 3. 绘制骨架点 (Scatter Plot)
        # *vertices.T 将数据转置为 [3, Num_Bodies]，分别解包给 x, y, z 参数
        # color="black": 黑色点
        # depthshade=False: 关闭深度阴影（防止远处的点变淡，保持所有点清晰可见）
        self._figure_axes.scatter(*vertices.T, color="black", depthshade=False)

        # 4. 调整坐标轴视野 (Camera/Axes Limits)
        
        # --- 模式 A: 场景视图 (Scene View) ---
        # 视野范围基于“所有帧”的最大最小值计算。
        # 效果：镜头固定不动，你能看到机器人从屏幕一端走到另一端。
        if self._render_scene:
            # 计算整个动作序列的包围盒 (Bounding Box)
            minimum = np.min(self._body_positions.reshape(-1, 3), axis=0)
            maximum = np.max(self._body_positions.reshape(-1, 3), axis=0)
            center = 0.5 * (maximum + minimum)
            diff = 0.75 * (maximum - minimum) # 留出 25% 的边缘空白

        # --- 模式 B: 骨架视图 (Skeleton View) ---
        # 视野范围基于“当前帧”的最大最小值计算。
        # 效果：镜头会一直由中心对准机器人（Camera Follow），机器人看起来是在原地运动。
        else:
            # 计算当前帧骨架的包围盒
            minimum = np.min(vertices, axis=0)
            maximum = np.max(vertices, axis=0)
            center = 0.5 * (maximum + minimum)
            # diff 取三轴中最大的跨度，确保 XYZ 轴比例一致 (Aspect Ratio = 1)
            diff = np.array([0.75 * np.max(maximum - minimum).item()] * 3)

        # 5. 应用坐标轴限制
        self._figure_axes.set_xlim((center[0] - diff[0], center[0] + diff[0]))
        self._figure_axes.set_ylim((center[1] - diff[1], center[1] + diff[1]))
        self._figure_axes.set_zlim((center[2] - diff[2], center[2] + diff[2]))
        # 强制设置坐标轴比例一致，防止图形被拉伸变形
        self._figure_axes.set_box_aspect(aspect=diff / diff[0])

        # 6. 绘制地面 (Ground Plane)
        # 在 Z=0 处画一个绿色的半透明平面，代表地面
        x, y = np.meshgrid([center[0] - diff[0], center[0] + diff[0]], [center[1] - diff[1], center[1] + diff[1]])
        self._figure_axes.plot_surface(x, y, np.zeros_like(x), color="green", alpha=0.2)

        # 7. 设置标签和标题
        self._figure_axes.set_xlabel("X")
        self._figure_axes.set_ylabel("Y")
        self._figure_axes.set_zlabel("Z")
        self._figure_axes.set_title(f"frame: {self._current_frame}/{self._num_frames}")

        # 8. 更新帧计数器
        self._current_frame += 1
        # 如果播放完所有帧，循环回到第 0 帧
        if self._current_frame >= self._num_frames:
            self._current_frame = 0

```

### 3.4 启动动画 (`show`)

```python
    def show(self) -> None:
        """显示动作动画窗口"""
        # 1. 创建 Matplotlib 画布
        self._figure = plt.figure()
        # 添加 3D 子图
        self._figure_axes = self._figure.add_subplot(projection="3d")

        # 2. 创建动画实例
        # fig: 画布对象
        # func: 每一帧调用的回调函数
        # frames: 总帧数 (决定动画循环长度)
        # interval: 帧间隔时间 (毫秒)。dt 是秒，所以要 * 1000。
        #           这保证了动画播放速度与真实动作速度一致。
        self._animation = matplotlib.animation.FuncAnimation(
            fig=self._figure,
            func=self._drawing_callback,
            frames=self._num_frames,
            interval=1000 * self._motion_loader.dt,
        )
        
        # 3. 阻塞主线程，显示窗口
        plt.show()

```

### 3.5 主程序入口 (`__main__`)

用于通过命令行直接运行此脚本。

```python
if __name__ == "__main__":
    import argparse

    # 1. 定义命令行参数
    parser = argparse.ArgumentParser()
    # --file: 必填，动作文件路径
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    # --render-scene: 可选，开关场景渲染模式
    parser.add_argument(
        "--render-scene",
        action="store_true", # 如果命令行加了这个参数，值为 True，否则为 False
        default=False,
        help=(
            "Whether the scene (space occupied by the skeleton during movement) is rendered instead of a reduced view"
            " of the skeleton."
        ),
    )
    # --matplotlib-backend: 可选，指定后端 (如 TkAgg, Qt5Agg)
    # 有时候默认后端在服务器或特定环境下无法弹窗，需要手动指定
    parser.add_argument("--matplotlib-backend", type=str, default="TkAgg", help="Matplotlib interactive backend")
    
    # 解析参数 (parse_known_args 允许忽略未定义的参数，方便与其他工具集成)
    args, _ = parser.parse_known_args()

    # 2. 设置 Matplotlib 后端
    # 参考: https://matplotlib.org/stable/users/explain/figure/backends.html#interactive-backends
    matplotlib.use(args.matplotlib_backend)

    # 3. 实例化并显示
    viewer = MotionViewer(args.file, render_scene=args.render_scene)
    viewer.show()

```