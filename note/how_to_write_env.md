# 📝 Isaac Lab 环境配置核心指南 (Manager-Based)

在 Isaac Lab 中，一个标准的 RL 环境配置文件通常继承自 `ManagerBasedRLEnvCfg`，其核心思想是 **“物理场景 (Scene)” 与 “逻辑决策 (MDP)” 分离**。

## 1. 场景定义 (Scene Definition)

**类名示例**: `MySceneCfg(InteractiveSceneCfg)`
**作用**: 定义物理世界里有什么东西。不涉及强化学习逻辑，只负责“摆东西”。

* **Terrain (地形)**: 地面怎么生成？
* `TerrainImporterCfg`: 导入平面或生成崎岖地形 (`terrain_type="generator"`).
* 关键参数: `physics_material` (摩擦力), `visual_material` (纹理).


* **Robot (机器人)**: 主角是谁？
* 通常引用预定义的 `ArticulationCfg` (如 `ANYMAL_C_CFG`).
* `prim_path="{ENV_REGEX_NS}/Robot"`: 正则路径，确保并行环境时名字不冲突.


* **Sensors (传感器)**: 机器人身上带了什么？
* `RayCasterCfg`: 激光雷达/高度扫描 (用于感知地形).
* `ContactSensorCfg`: 接触传感器 (判断脚是否着地).


* **Lights (灯光)**: `DomeLight`, `DistantLight`.

```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(...)
    robot: ArticulationCfg = MISSING # 留空，在具体类中填充
    height_scanner = RayCasterCfg(...) # 高度扫描
    contact_forces = ContactSensorCfg(...) # 足底接触

```

---

## 2. 命令 (Commands)

**类名示例**: `CommandsCfg`
**作用**: 定义机器人的任务目标（Task）。

* **Command Type**: 比如 `UniformVelocityCommandCfg` (速度追踪任务).
* **Ranges**: 目标的随机范围.
* `lin_vel_x`: 前后速度范围 (-1.0 到 1.0 m/s).
* `heading`: 偏航角目标.


* **Resampling**: 多久换一次命令 (`resampling_time_range`).

---

## 3. 动作 (Actions)

**类名示例**: `ActionsCfg`
**作用**: 定义神经网络输出层 (Output) 如何控制机器人。

* **Control Mode**:
* `JointPositionActionCfg`: 位置控制 (PD Control).
* `JointEffortActionCfg`: 力矩控制.


* **Scale**: 缩放系数. 神经网络输出通常在 [-1, 1], 需要缩放到实际物理单位 (比如弧度或牛顿).

---

## 4. 观测 (Observations)

**类名示例**: `ObservationsCfg`
**作用**: 定义神经网络输入层 (Input)。

* **Policy Group**: 给 Actor/Critic 网络看的输入.
* **Observation Terms**:
* `base_lin_vel`, `base_ang_vel`: 基座速度.
* `joint_pos`, `joint_vel`: 关节状态.
* `actions`: 上一帧的动作 (用于记忆).
* `height_scan`: 地形感知信息.


* **Noise (关键)**: **Sim-to-Real 的核心**.
* `noise=Unoise(n_min=-0.1, n_max=0.1)`: 在仿真数据上叠加高斯或均匀噪声，模拟真实传感器的误差.



---

## 5. 事件/域随机化 (Events / Domain Randomization)

**类名示例**: `EventCfg`
**作用**: 让环境“变着花样”折磨 AI，提高鲁棒性。

* **Startup (启动时)**:
* `randomize_rigid_body_mass`: 随机改质量 (模拟负载变化).
* `randomize_rigid_body_material`: 随机改摩擦力 (模拟不同地面).


* **Reset (重置时)**:
* `reset_root_state_uniform`: 随机初始位置和朝向.
* `reset_joints_by_scale`: 随机初始关节角度.


* **Interval (定时间隔)**:
* `push_robot`: 每隔几秒推一下机器人 (学习抗干扰能力).



---

## 6. 奖励函数 (Rewards)

**类名示例**: `RewardsCfg`
**作用**: 告诉 AI 什么是对的 (Refinement).

* **Task Rewards (正分)**:
* `track_lin_vel_xy_exp`: 跟踪目标速度，越准分越高.


* **Penalties (负分)**:
* `lin_vel_z_l2`: 惩罚上下颠簸 (z轴速度).
* `action_rate_l2`: 惩罚动作剧烈突变 (保护电机).
* `dof_torques_l2`: 惩罚能量消耗.
* `undesired_contacts`: 惩罚大腿/身体撞地.



---

## 7. 终止条件 (Terminations)

**类名示例**: `TerminationsCfg`
**作用**: 什么时候 Game Over 并重置 (Reset)。

* `time_out`: 超时 (例如跑了 20秒).
* `base_contact`: 摔倒了 (身体基座接触地面).

---

## 8. 组装与后处理 (Assembly & Post-init)

**类名示例**: `LocomotionVelocityRoughEnvCfg`

这是最终的组装类，继承自 `ManagerBasedRLEnvCfg`。

### 核心参数 (`__post_init__`):

1. **`decimation` (抽帧/控制频率)**:
* 物理引擎步长 `sim.dt` (例如 0.005s = 200Hz).
* `decimation = 4`.
* **控制频率** = 200Hz / 4 = **50Hz**.


2. **`episode_length_s`**: 一局多长 (20秒).
3. **`num_envs`**: 并行环境数量 (4096个).

### 机器人特定配置 (`AnymalCRoughEnvCfg`):

* 通过继承通用配置类。
* 使用 `.replace()` 方法替换 `scene.robot` 为具体的机器人 (如 ANYmal-C).
* 如果是 **Play** 模式 (推理/演示)，通常会:
* 减少环境数 (`num_envs=50`).
* **关闭随机化** (`enable_corruption=False`, `push_robot=None`).



---

## 💡 总结图示 (Workflow)

```mermaid
graph TD
    A[EnvCfg (总配置)] --> B[SceneCfg (物理场景)]
    A --> C[MDP Settings (逻辑与算法)]
    
    B --> B1[Terrain 地形]
    B --> B2[Robot 机器人]
    B --> B3[Sensors 传感器]
    
    C --> C1[Observations 输入]
    C --> C2[Actions 输出]
    C --> C3[Rewards 奖励函数]
    C --> C4[Commands 任务目标]
    C --> C5[Events 随机化]

```