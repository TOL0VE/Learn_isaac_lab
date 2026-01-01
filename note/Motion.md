# 动作文件 (Motion files)

动作文件采用 NumPy 文件格式，其中包含执行该动作的骨架自由度 (DOF) 和刚体的数据。

文件中的数据（通过键名访问）如下表所述，其中：

* `N` 代表记录的动作帧数
* `D` 代表骨架自由度 (DOF) 的数量
* `B` 代表骨架刚体 (Body) 的数量

| 键名 (Key) | 数据类型 (Dtype) | 形状 (Shape) | 描述 (Description) |
| --- | --- | --- | --- |
| `fps` | int64 | () | 动作采样的帧率 (FPS) |
| `dof_names` | unicode string | (D,) | 骨架自由度 (DOF) 名称 |
| `body_names` | unicode string | (B,) | 骨架刚体 (Body) 名称 |
| `dof_positions` | float32 | (N, D) | 骨架自由度位置 (角度/位移) |
| `dof_velocities` | float32 | (N, D) | 骨架自由度速度 |
| `body_positions` | float32 | (N, B, 3) | 骨架刚体位置 |
| `body_rotations` | float32 | (N, B, 4) | 骨架刚体旋转 (四元数格式 `wxyz`) |
| `body_linear_velocities` | float32 | (N, B, 3) | 骨架刚体线速度 |
| `body_angular_velocities` | float32 | (N, B, 3) | 骨架刚体角速度 |

## 动作可视化 (Motion visualization)

`motion_viewer.py` 文件允许你可视化记录在动作文件中的骨架运动。

请在 `motions` 文件夹中打开终端，并运行以下命令：

```bash
python motion_viewer.py --file MOTION_FILE_NAME.npz

```

使用 `python motion_viewer.py --help` 命令可查看所有可用参数。