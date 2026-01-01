./isaaclab.sh --new
# 1.define a roobot
>
>source/isaaclab_assets/isaaclab_assets/robots/cartpole.py

```python
"""Configuration for a simple Cartpole robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

CARTPOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 1. 3D模型路径：去 NVIDIA 的云端服务器 (Nucleus) 下载 cartpole.usd 文件
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
        #如果你想用自己本地魔改的 USD 文件，可以直接写绝对路径 "/home/user/my_robot.usd"。
        
        # 2. 刚体属性：限制最大速度，防止仿真炸飞
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0, # 限制最大线速度
            enable_gyroscopic_forces=True, # 开启陀螺效应计算（让物理更真实）
        ),
        
        # 3. 关节求解器属性：给物理引擎看的参数
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, # 比如杆子会不会穿过底座？False表示不检测自碰撞
            solver_position_iteration_count=4, # 物理计算精度，通常 4~8 够用了
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(x, y, z): 出生在半空中 2.0 米的位置（这样它可能会掉下来砸到地上）
        pos=(0.0, 0.0, 2.0), 
        
        # 关节初始角度：所有关节归零，杆子是直的
        joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
    ),
    actuators={
        # --- 小车的电机 (cart_actuator) ---
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"], # 控制哪个关节？控制滑块
            effort_limit_sim=400.0,              # 最大推力：400牛
            stiffness=0.0,                       # 刚度 P gain：0 (力控模式)
            damping=10.0,                        # 阻尼 D gain：10 (模拟摩擦力/反电动势)
        ),
        
        # --- 杆子的关节 (pole_actuator) ---
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],   # 控制哪个关节？连接杆子的关节
            effort_limit_sim=400.0,
            stiffness=0.0, 
            damping=0.0                          # 阻尼是 0！这意味着这是一个【无摩擦摆】
        ),
    },
)
"""Configuration for a simple Cartpole robot."""
```

* complicated example

```python
"""Configuration for the 28-DOFs Mujoco Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

HUMANOID_28_CFG = ArticulationCfg(
    # prim_path 指定该资产在 USD 舞台中的路径，{ENV_REGEX_NS} 会被替换为环境编号（如 /World/envs/env_0）
    prim_path="{ENV_REGEX_NS}/Robot",
    
    # spawn 定义如何生成这个资产（从 USD 文件加载）
    spawn=sim_utils.UsdFileCfg(
        # 资产文件的路径，指向 Isaac Nucleus 服务器上的标准 Humanoid 资产
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Humanoid28/humanoid_28.usd",
        
        # 刚体物理属性配置
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,             # None 表示使用物理场景的默认重力设置
            
            # 最大去穿透速度：当两个物体穿模时，物理引擎把它们推开的最大速度。
            # 设置为 10.0 较大，意味着允许较剧烈的反弹，防止穿模卡死，但也可能导致抖动。
            max_depenetration_velocity=10.0,  
            
            # 开启陀螺效应力：对于人形机器人这种高速旋转部件较少但整体平衡关键的系统，
            # 开启它能更精确地模拟角动量守恒（例如空翻时的姿态保持）。
            enable_gyroscopic_forces=True,
        ),
        
        # 关节系统属性配置
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 开启自碰撞：机器人的手不能穿过自己的身体。这对强化学习很重要，防止学会“穿模”这种作弊动作。
            enabled_self_collisions=True,
            
            # 求解器迭代次数：位置迭代 4 次，速度迭代 0 次。
            # 这是一个较低的设置（通常精度高需要 8+），这里为了追求极快的训练速度而牺牲了一些物理精度。
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            
            # 休眠阈值：当机器人几乎不动时让其“休眠”以节省算力。
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    
    # 初始状态配置
    init_state=ArticulationCfg.InitialStateCfg(
        # 初始位置：x=0, y=0, z=0.8米。
        # 让机器人从半空中掉下来，避免初始化时脚陷在地里导致物理爆炸。
        pos=(0.0, 0.0, 0.8),
        # 所有关节角度初始化为 0（通常是“大字形”或直立姿态）。
        joint_pos={".*": 0.0},
    ),
    
    # 驱动器配置：这是最重要的部分！
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"], # 匹配所有关节
            
            # 刚度(P)和阻尼(D)都设为 None，意味着这是一个“纯力矩控制”模式。
            # 动作输出 = 扭矩。没有 PD 控制器的介入。
            # 这符合经典 MuJoCo 环境的设定，控制难度较高。
            stiffness=None,
            damping=None,
            
            # 关节速度限制：100 rad/s，非常宽松，几乎不限制。
            velocity_limit_sim={".*": 100.0},
        ),
    },
)
```

```python
ANYMAL_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        
        # 开启接触传感器：对于四足机器人，知道哪只脚着地对于步态规划至关重要。
        # 开启后，可以通过 API 直接获取足端接触力。
        activate_contact_sensors=True,
        
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            
            # 不保留加速度：上一帧的加速度不会传到下一帧。
            # 这有助于在重置环境（reset）时消除伪影，但在连续仿真中略微不符合物理。
            retain_accelerations=False,
            
            # 线性阻尼和角阻尼设为 0：
            # 意味着没有“空气阻力”。能量耗散完全依赖于接触摩擦和电机内部的阻尼，这更符合真实物理建模。
            linear_damping=0.0,
            angular_damping=0.0,
            
            # 限制最大速度：防止仿真爆炸（出现 NaN 或飞出天际）。
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            
            # 这里设为 1.0，比人形机器人的 10.0 小很多。
            # 说明这个模型追求更精确的接触处理，不允许剧烈的穿透反弹。
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    
    # 初始状态配置：非常具体的姿态初始化
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6), # 初始高度 0.6米
        joint_pos={
            # 正则表达式匹配关节名：
            ".*HAA": 0.0,   # HAA (Hip Abduction Adduction) 髋关节侧摆：0度
            ".*F_HFE": 0.4,  # Front HFE (Hip Flexion Extension) 前腿髋关节前后摆：0.4弧度（前腿向后弯）
            ".*H_HFE": -0.4, # Hind HFE 后腿髋关节前后摆：-0.4弧度（后腿向前弯）
            ".*F_KFE": -0.8, # Front KFE (Knee) 前腿膝关节：-0.8弧度
            ".*H_KFE": 0.8,  # Hind KFE 后腿膝关节：0.8弧度
            # 这种“X”型或“< >”型的腿部姿态是四足机器人标准的站立初始位，防止一开始就跪在地上。
        },
    ),
    
    # 驱动器配置：高级黑科技
    # 这里引用了一个外部变量 ANYDRIVE_3_LSTM_ACTUATOR_CFG。
    # 这是一个 Actuator Net（执行器网络），它不是简单的 PD 公式，
    # 而是一个训练好的 LSTM 神经网络，能够模拟真实电机的延迟、饱和、摩擦等非线性特性。
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    
    # 软关节限位系数：0.95
    # 当关节接近物理极限的 95% 时，就开始施加反向力阻止其继续运动，
    # 保护关节不会因为冲过头而使得物理引擎报错。
    soft_joint_pos_limit_factor=0.95,
)
```

# 2.define a env

>/home/oiioaa/Desktop/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py

## 2.1 manager_based (only config)

### 2.1.1 dependencies

```python
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Scene definition
##
```

### 2.1.2 scene

* InteractiveSceneCfg是所有 RL 环境的基类，它会自动处理场景的重置（Reset）和克隆（Cloning）。

```python
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    # 👇 看这里！这个变量名就叫 robot
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    #prim_path在 Isaac Sim (以及底层的 USD 格式) 中，它的意思就是：“这个物体在 3D 世界里的【绝对地址】”。
    #prim_path="{ENV_REGEX_NS}/Robot"
    '''
    实际运行时的样子：
    第 0 号环境：/World/envs/env_0/Robot
    第 1 号环境：/World/envs/env_1/Robot
    第 2 号环境：/World/envs/env_2/Robot
    ...
    第 4095 号环境：/World/envs/env_4095/Robot
    '''

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
```

### 2.1.3 action

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    #AI 输出的数字，要被当作【推力】(Effort)，施加在小车的【滑轨关节】上，并且要【放大100倍】

```

### 2.1.4 obs(Actor&Critic)

```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    # --- 给 Actor 看的 (有限信息) ---
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        #cancelled joint_vel_rel to reduce input size.test lstm performance
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    # 老师和学生看的一样，所以省略 CriticCfg
    '''
    example
    class CriticCfg(ObsGroup):
        sensors = ObsTerm(func=...) # 无噪声
        height_map = ObsTerm(func=...) # 地图
        ground_friction = ObsTerm(func=...) # 摩擦力
    critic: CriticCfg = CriticCfg()
    '''
```

### 2.1.5 event(random reset)

```python
@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )
```

### 2.1.6 reward

```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},#,<-变量名 (Argument Name) 必须和 params 里的键名（Key）一字不差。
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

### 2.6.1 custom reward

* build a private dir(mdp) to writte custom reward

```python
from __future__ import annotations#“请忽略所有的类型提示（Type Hints），不要在运行时去检查它们存不存在。留给 IDE 和静态检查工具去看就行了。”

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

                                               #👇变量名 (Argument Name) 必须和 params 里的键名（Key）一字不差。
def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

```

### 2.1.7 terminate

```python
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    '''
    含义：如果这一局跑了太久（比如超过了 500 步），强制结束。
    func=mdp.time_out：调用官方通用的计时检查函数。它会检查当前的步数（episode length）是否达到了 max_episodes（在主脚本里设置，比如 500）。
    time_out=True：这是一个非常重要的标志位！
    它告诉算法：“这局结束不是因为机器人太菜（输了），而是因为没时间了。”
    数学意义：在算奖励价值（Value Function）时，超时结束通常不会把未来的预期奖励归零（Bootstrap），因为它其实还能继续活下去；而失败结束（撞墙）则会归零。
    '''
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )
```

>call back

```python
class RewardsCfg:
    ....
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    ...
```

* if want multireward

```python
@configclass
class RewardsCfg:
    # ... 其他奖励保持不变 ...

    # 1. 删除这个通用的！不然会重复扣分
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0) 

    # 2. 新增：专门针对“车出界”的罚款 (重罚)
    # 逻辑和 TerminationsCfg.cart_out_of_bounds 完全一样
    penalty_cart_out = RewTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        weight=-10.0,  # <--- 这里设置车出界扣 10 分
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "bounds": (-3.0, 3.0), # 必须和 Termination 里的范围一致
        },
    )

    # 3. 新增：专门针对“杆子倒了”的罚款 (轻罚)
    # 假设 Termination 里也有个类似的 pole_limit
    penalty_pole_drop = RewTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        weight=-5.0,   # <--- 这里设置杆子倒了扣 5 分
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "bounds": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )
```

### 2.1.8 total

```python
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5  #一局多长：5 秒。换算成步数：5 秒 X 60  Hz (控制频率) = 300  步。
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120 # 物理引擎的计算步长
        self.sim.render_interval = self.decimation # 决策频率倍数 (每隔几次物理步，AI 动一次脑子)

```

### 2.1.9 camera

```python
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

from .cartpole_env_cfg import CartpoleEnvCfg, CartpoleSceneCfg
```

#### 2.1.9.1 add camera(RGB&)

```python
@configclass
class CartpoleRGBCameraSceneCfg(CartpoleSceneCfg):

    # add camera to the scene
    #TiledCameraCfg(平铺渲染技术)
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        '''
        X = -7.0: 放在小车后方 7 米处（假设车头朝 X 正方向）。
        Y = 0.0: 左右居中。
        Z = 3.0: 悬空 3 米高。
        视角效果：这是一个典型的第三人称上帝视角 (God View)，类似于玩赛车游戏时的默认视角。

        (0.9945, ...)
        这是一个四元数，表示相机微微向下低头（Pitch 轴旋转），以便从 3 米高的地方正好俯视地面上的车。
        '''
        data_types=["rgb"], #<-
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        '''
        这定义了相机的光学属性，就像你买单反镜头时看的参数。
        focal_length=24.0 (焦距 24mm):
        这是一个广角镜头。视野比较宽，能看到更多的环境，但边缘会有一些透视拉伸。
        horizontal_aperture=20.955 (传感器宽度):
        配合焦距，这决定了相机的 视场角 (FOV)。
        clipping_range=(0.1, 20.0) (视距裁剪):
        近裁剪 (0.1): 离镜头小于 0.1 米的物体不渲染（防止穿模挡住镜头）。
        远裁剪 (20.0): 离镜头超过 20 米的物体不渲染（直接显示背景色）。这能极大节省显卡资源，反正太远了 AI 也看不清。
        '''
        width=100,
        height=100,
    )


@configclass
class CartpoleDepthCameraSceneCfg(CartpoleSceneCfg):

    # add camera to the scene
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["distance_to_camera"], #<-
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )

```

#### 2.1.9.2 obs

```python
@configclass
class RGBObservationsCfg:
    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        # 1. 定义输入源
        image = ObsTerm(
            func=mdp.image,  # <-- 这是一个获取原始像素的函数
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"), # 告诉它去读哪个相机
                "data_type": "rgb" # 告诉它读红绿蓝三个通道
            }
        )

        def __post_init__(self):
            self.enable_corruption = False # 不加人为噪声(如坏点)
            self.concatenate_terms = True  # 把数据拼成一个 Tensor

    policy: ObsGroup = RGBCameraPolicyCfg()


@configclass
class DepthObservationsCfg:
    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.image, 
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"), 
                "data_type": "distance_to_camera" # <-- 关键区别：读取距离
            }
        )

    policy: ObsGroup = DepthCameraPolicyCfg()


@configclass
class ResNet18ObservationCfg:
    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.image_features, # <-- 关键区别：不再取图片，而是取特征
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"), 
                "data_type": "rgb", 
                "model_name": "resnet18" # <-- 请来了外援：ResNet18
            },
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


@configclass
class TheiaTinyObservationCfg:
    @configclass
    class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cddsv", # <-- 这是一个 Transformer 模型
                "model_device": "cuda:0", # 指定模型跑在 GPU 上
            },
        )

    policy: ObsGroup = TheiaTinyFeaturesCameraPolicyCfg()
```

#### 2.1.9.3 env

```python
@configclass
class CartpoleRGBCameraEnvCfg(CartpoleEnvCfg):
    """Configuration for the cartpole environment with RGB camera."""
    
    # 1. 场景配置：不仅仅是换个名字，注意参数的变化！
    #env_spacing=20如果间距只有 4 米：0号环境的相机，会直接架在 1号环境的家里！ 0号相机会拍到 1号小车的屁股，而不是 0号小车。所以必须把环境拉开到 20 米远，保证每个相机只能看到自己家的小车。
    scene: CartpoleRGBCameraSceneCfg = CartpoleRGBCameraSceneCfg(num_envs=512, env_spacing=20)
    
    # 2. 观测配置：指定用 RGB 像素作为输入
    observations: RGBObservationsCfg = RGBObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 3. 把地板拆了
        self.scene.ground = None
        
        # 4. 调整人类观察者的视角
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)


@configclass
class CartpoleDepthCameraEnvCfg(CartpoleEnvCfg):
    """Configuration for the cartpole environment with depth camera."""

    scene: CartpoleDepthCameraSceneCfg = CartpoleDepthCameraSceneCfg(num_envs=512, env_spacing=20)

    # 只改了一行！
    observations: DepthObservationsCfg = DepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None
        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)


@configclass
class CartpoleResNet18CameraEnvCfg(CartpoleRGBCameraEnvCfg):
    """Configuration for the cartpole environment with ResNet18 features as observations."""
    # 只改了一行！
    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()


@configclass
class CartpoleTheiaTinyCameraEnvCfg(CartpoleRGBCameraEnvCfg):
    """Configuration for the cartpole environment with Theia-Tiny features as observations."""
    # 只改了一行！
    observations: TheiaTinyObservationCfg = TheiaTinyObservationCfg()
```

## 2.2 direct

### 2.2.1 config

```python
from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10.0  # 每个 episode 最长 10 秒
    decimation = 2           # 抽帧率。控制频率 = 物理频率 / 2。
                             # 如下文 sim.dt=1/60s (60Hz)，则控制频率为 30Hz。

    # spaces
    observation_space = 81       # 策略网络(Actor)的观测维度
    action_space = 28            # 动作维度（对应28个自由度）
    state_space = 0              # 这里的 state 通常指非对称 Critic 的输入，Direct 模式下有时不显式使用
    
    # AMP 特有参数
    num_amp_observations = 2     # AMP 需要存储最近 2 帧的观测历史
    amp_observation_space = 81   # 判别器(Discriminator)的观测维度，通常比 Actor 的观测更精简（去掉了目标位置等）

    early_termination = True     # 开启提前终止（比如摔倒就重置）
    termination_height = 0.5     # 摔倒判定阈值：如果躯干高度低于 0.5米，视为摔倒

    motion_file: str = MISSING   # 占位符，具体的动作文件路径由子类填入
    reference_body = "torso"     # 模仿动作时的参考核心身体部件
    reset_strategy = "random"    # 重置策略：这是 AMP 的关键！
    """
    * default: 回到 T-pose 或初始站立姿态。
    * random: (AMP常用) 从动作数据集中随机采样一个时间点，把机器人摆成那个姿势开始。
              这让机器人学会从各种中间动作继续运动，而不仅仅是从头开始。
    * random-start: 从动作的第0秒开始。
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,                   # 物理引擎步长 60Hz
        render_interval=decimation,  # 渲染间隔，与控制频率对齐
        physx=PhysxCfg(
            # 显存优化参数：增加 GPU 上的碰撞对缓冲区容量
            # 对于 Humanoid 这种多关节且环境数量巨大(4096)的场景，这能防止显存溢出导致的物理崩溃
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,           # 并行环境数量：4096个。Direct模式极其高效，可以支持大规模并行。
        env_spacing=10.0,        # 环境间距
        replicate_physics=True   # 开启物理复制（PhysX 高级特性），加速大规模同构环境
    )

    # 机器人配置
    # 这里直接复用了你之前看过的 HUMANOID_28_CFG
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"  # 正则表达式，确保每个环境里的机器人都有唯一路径
    ).replace(
        # 再次确认驱动器配置：
        # 这里显式覆盖了一次，确保使用的是无刚度、无阻尼的纯力矩控制
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,  # P = 0
                damping=None,    # D = 0
                velocity_limit_sim={".*": 100.0},
            ),
        },
    )
@configclass
class HumanoidAmpDanceEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")

```

### 2.2.2 env (implementation)

**Example Path:** `source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/humanoid_amp_env.py`

在 **Direct (直接)** 工作流中，环境逻辑不是通过配置文件组合出来的，而是通过继承 `DirectRLEnv` 并重写核心方法（如 `_get_observations`, `_apply_action` 等）来手动实现的。这通常用于追求极致性能或实现非标准逻辑（如 AMP 算法）。

#### 2.2.2.1 Imports & Class Definition (导入与类定义)

```python
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg
from .motions import MotionLoader

# 继承自 DirectRLEnv (代码驱动环境的基类)
class HumanoidAmpEnv(DirectRLEnv):
    # 类型标注：告诉 IDE 这个 cfg 是 HumanoidAmpEnvCfg 类型的
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        # 初始化父类
        super().__init__(cfg, render_mode, **kwargs)

        # --- 1. 预计算动作缩放参数 (Action Scaling) ---
        # Direct 模式需要手动实现动作到物理量的映射逻辑。
        # 这里计算了关节软限位，以便将神经网络输出的 [-1, 1] 映射到 [min_joint_pos, max_joint_pos]。
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits) # 中点
        self.action_scale = dof_upper_limits - dof_lower_limits          # 范围

        # --- 2. 加载参考动作 (Motion Loader) ---
        # AMP 算法的核心：加载专家数据（如人类动作捕捉数据 .npz 文件）
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # --- 3. 缓存关键身体部件索引 (Indexing) ---
        # 在仿真循环中查字典太慢了，所以要在初始化时把身体部件的名字（如"right_hand"）
        # 转换成数字索引，方便后续快速查表。
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body) # 核心参考点（通常是躯干）
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        
        # 获取动作数据中对应的关节索引，确保仿真模型的关节顺序和动作文件的顺序一致
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # --- 4. 配置 AMP 专用观测空间 (AMP Observations) ---
        # AMP 判别器需要查看“历史动作片段”。
        # 这里的 amp_observation_buffer 就是显存里的一块专用区域，用来存过去几帧的观测数据。
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

```

#### 2.2.2.2 _setup_scene (构建物理场景)

在 Manager-Based 模式里这是自动的，但在 Direct 模式里，你必须亲手把东西“放”进世界里。

```python
    def _setup_scene(self):
        # 1. 创建机器人对象 (Articulation)
        # 根据配置生成 Articulation 实例，负责管理机器人的物理状态（关节、刚体等）
        self.robot = Articulation(self.cfg.robot)
        
        # 2. 手动创建地面 (Ground Plane)
        # 定义地面的摩擦力等物理属性
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0, # 无弹性，掉地上不会弹起来
                ),
            ),
        )
        
        # 3. 克隆环境 (Cloning) - ⚠️ 关键步骤
        # Isaac Sim 需要把上面定义的这一套环境复制 4096 份（并行训练）。
        # Manager-Based 自动做，这里要显式调用。
        self.scene.clone_environments(copy_from_source=False)
        
        # 如果是 CPU 模式，需要手动过滤碰撞（GPU 模式通常由 PhysX 处理）
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # 4. 注册机器人到场景
        # 只有注册了，scene.step() 才会更新这个机器人的状态
        self.scene.articulations["robot"] = self.robot
        
        # 5. 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

```

#### 2.2.2.3 _apply_action (执行动作)

定义如何将神经网络输出的 action 转化为物理引擎的指令。

```python
    def _pre_physics_step(self, actions: torch.Tensor):
        # 缓存动作，防止被修改
        self.actions = actions.clone()

    def _apply_action(self):
        # 1. 动作映射 (Action Mapping)
        # 使用简单的线性变换：目标位置 = 偏移量 + 缩放 * 动作
        # 这相当于一个 P-Controller 的设定值
        target = self.action_offset + self.action_scale * self.actions
        
        # 2. 下发指令 (Command)
        # 直接调用底层 API 设置关节的目标位置
        # 因为在 cfg 里我们将刚度/阻尼设为了 None (力矩模式)，
        # 但这里设定位置目标通常意味着我们实际上是在用仿真器内部的驱动器求解器，
        # 或者这其实是一个位置控制的任务配置（取决于 robot cfg 的具体写法，但在 AMP 中通常是 PD 控制）。
        self.robot.set_joint_position_target(target)

```

#### 2.2.2.4 _get_observations (计算观测)

这是 Direct 模式最核心的优势：使用 JIT 编译加速观测计算。

```python
    def _get_observations(self) -> dict:
        # 1. 计算当前帧观测 (Task Obs)
        # 调用下面那个 @torch.jit.script 加速的函数 compute_obs
        # 一次性传入所有环境（4096个）的关节位置、速度、根节点状态等数据
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # 2. 更新 AMP 历史缓冲区 (History Buffer)
        # 这是一个 FIFO 队列操作：把旧数据往后移，把新数据填入最前面
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        
        # 3. 填入最新观测
        self.amp_observation_buffer[:, 0] = obs.clone()
        
        # 4. 返回结果
        # "policy": 给 Actor 网络的输入
        # "extras": 额外信息，这里把 AMP buffer 塞进去，供判别器 (Discriminator) 使用
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

```

#### 2.2.2.5 _get_rewards & _get_dones (奖励与终止)

```python
    def _get_rewards(self) -> torch.Tensor:
        # ⚠️ 注意：AMP 任务的奖励特殊性
        # 在 AMP 中，"风格奖励 (Style Reward)" 是由判别器计算的，不在环境里算。
        # 环境只负责给一个 "生存奖励 (Survival Reward)"，即只要活着就给 1 分。
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. 超时判断 (Time Out)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 2. 死亡判断 (Died)
        # 如果启用了提前终止，检查躯干高度是否低于阈值（例如摔倒了）
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
            
        return died, time_out

```

#### 2.2.2.6 _reset_idx (重置逻辑)

AMP 的训练非常依赖“随机重置”，即从专家的动作片段中随机选一个姿态开始训练，而不是每次都从站立开始。

```python
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        
        # 1. 重置机器人物理状态
        self.robot.reset(env_ids)
        
        # 2. 调用父类重置（清理计时器等）
        super()._reset_idx(env_ids)

        # 3. 选择重置策略
        if self.cfg.reset_strategy == "default":
            # 默认：回到初始 T-pose
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            # 随机：从动作库里采样一个姿态 (这是 AMP 训练成功的关键！)
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        # 4. 强制写入仿真器 (Write to Sim)
        # 将计算好的 根节点位置、速度、关节角度 强行写入物理引擎，瞬间“瞬移”机器人
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # --- 具体的重置策略实现 ---

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 从 motion_loader 中采样随机时间点
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        
        # 从动作文件中读取该时间点的所有物理数据
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # 计算 Root State (躯干状态)
        motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        # 加上环境偏移量 (env_origins)，保证机器人落在自己的格子里
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.15  # 稍微抬高一点，防止脚插地里炸飞
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        # ... 设置速度 ...

        # 获取关节状态
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # ⚠️ 重要：重置 AMP 观测历史
        # 既然机器人瞬移到了新状态，它的“历史动作”也必须伪造为专家数据的历史，
        # 否则判别器会发现“前一秒还在别处，后一秒瞬移了”，从而给出极低分。
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

```

#### 2.2.2.7 JIT Accelerated Functions (JIT 加速函数)

为了在 GPU 上全速运行，复杂的数学计算被提出来并用 `@torch.jit.script` 装饰。这使得 PyTorch 可以将其编译为高效的 CUDA Kernel，避免 Python 循环开销。

```python
@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    """辅助函数：将四元数转换为切线和法线向量（对神经网络更友好的旋转表示）。"""
    # ... 数学计算 ...
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    # ... 其他输入 ...
) -> torch.Tensor:
    """
    观测计算核心函数。
    输入：所有并行的物理状态张量 (Tensor)
    输出：拼接好的观测张量
    """
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # 只取高度 (Z轴)
            quaternion_to_tangent_and_normal(root_rotations), # 旋转转换
            root_linear_velocities,
            root_angular_velocities,
            # 计算关键身体部件相对于根节点的相对位置
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs

```

# 3 define Runner

> /home/oiioaa/Desktop/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/agents/rsl_rl_ppo_cfg.py

```python
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

import isaaclab_tasks.manager_based.classic.cartpole.mdp.symmetry as symmetry
```

## 3.1 RSL_RL

### 3.1.1 MLP

```python
@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

```

### 3.1.2 LSTM+MLP

```python
@configclass
class MyLSTMPolicyCfg(RslRlPpoActorCriticCfg):
    rnn_type: str = "lstm"
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    class_name = "ActorCriticRecurrent"  #care

# Runner 配置 1
@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_lstm"
    empirical_normalization = False

    # 使用你的 LSTM 配置
    policy = MyLSTMPolicyCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# Runner 配置 2
# AI 利用物理对称性，数据翻倍，收敛更快。
@configclass
class CartpolePPORunnerWithSymmetryCfg(CartpolePPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    # all the other settings are inherited from the parent class
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=symmetry.compute_symmetric_states
        ),
    )

```

## 3.2 SKRL

### 3.2.1 seed

```yaml
seed: 24 #manba out
```

### 3.2.2 models

### 3.2.2.1 MLP

```yaml
models:
    separate: False  # 暂时忽略，意思是 Actor 和 Critic 定义在一起管理
  
    # === 策略网络 (Actor) ===
    policy:  
        class: GaussianMixin   # 使用高斯分布模型（因为要加随机噪声探索）
        
        # 下面这些 clip_log_std 等等，是直接传给 GaussianMixin __init__ 函数的参数
        # 你在文档的 GaussianMixin 类参数里能找到它们
        clip_actions: False
        clip_log_std: True
        min_log_std: -20.0
        max_log_std: 2.0
        initial_log_std: 0.0

        # 这里开始定义神经网络结构
        network:
        - name: net
            input: OBSERVATIONS  # 输入层大小 = 传感器数据长度
            layers: [32, 32]     # 中间有两个隐藏层，每层 32 个神经元
            activations: elu     # 激活函数用 ELU
        output: ACTIONS          # 输出层大小 = 机器人关节数量

    # ===  (Critic) ===
    value:
        class: DeterministicMixin
        clip_actions: False
        network:
        - name: net
            input: OBSERVATIONS
            layers: [32, 32]
            activations: elu
        output: ONE
```

| YAML 关键字 | 含义 | 实际 Python 逻辑 |
| --- | --- | --- |
| **`OBSERVATIONS`** | 观测空间维度 | `env.observation_space.shape[0]` (比如 48) |
| **`ACTIONS`** | 动作空间维度 | `env.action_space.shape[0]` (比如 12) |
| **`ONE`** | 标量 (用于 Critic) | `1` (Critic 输出的是一个价值分数，不是动作向量) |
| **`STATES`** | 全局状态维度 | 用于非对称 Critic (Teacher)，包含特权信息 |

#### 3.2.2.2 LSTM + MLP (future)

```yaml
models:
    separate: False  # 暂时忽略，意思是 Actor 和 Critic 定义在一起管理
  
    # === 策略网络 (Actor) ===
    policy:  
        class: GaussianMixin   # 使用高斯分布模型（因为要加随机噪声探索）
        
        # 下面这些 clip_log_std 等等，是直接传给 GaussianMixin __init__ 函数的参数
        # 你在文档的 GaussianMixin 类参数里能找到它们
        clip_actions: False
        clip_log_std: True
        min_log_std: -20.0
        max_log_std: 2.0
        initial_log_std: 0.0

        # 这里开始定义神经网络结构
        network:
            # 第一层：LSTM (记忆模块)
            - name: memory_layer
                input: OBSERVATIONS
                type: LSTM
                num_layers: 1          # LSTM 层数 定义层数的参数叫 num_layers
                hidden_size: 256   
                
            # 第二层：MLP (决策/解码模块)
            - name: decision_layer
                type: MLP
                layers: [128, 64]      # MLP 这里依然叫 layers，且必须是列表
                activations: elu
        output: ACTIONS

    # ===  (Critic) ===
    value:
        class: DeterministicMixin
        clip_actions: False
        network:
        - name: net
            input: OBSERVATIONS
            layers: [32, 32]
            activations: elu
        output: ONE
```

### 3.2.3 memory

```yaml
memory:
  class: RandomMemory
  memory_size: -1  # automatically determined (same as agent:rollouts)
  #“给我准备一个支持随机存取的临时背包 (RandomMemory)，背包的大小 (memory_size) 只要刚好能装下 Agent 这一轮采集的数据 (-1) 就行了。反正学完这一轮就要倒掉的。”
```

### 3.2.4 agent

#### 3.2.4.1 PPO

```yaml
# PPO agent configuration
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html

agent:
  # 1. 算法核心类
  class: PPO  

  # 2. 采样与学习节奏 (The Loop)
  # 含义：每次更新网络前，每个环境先玩几步？
  # 计算：如果 num_envs=4096, rollouts=16，那总数据量 = 65,536 步
  # 建议：保持 16~24，这是 Isaac Lab 的经验值
  rollouts: 16

  # 含义：拿到的这批 65,536 条数据，要反复复习几遍？
  # 建议：5~8 遍。太少学不会，太多会过拟合（钻牛角尖）
  learning_epochs: 8

  # 含义：一口气吃不下 6万条数据，切成几块喂给显卡？
  # 建议：4 或 8。切得越细，显存占用越低
  mini_batches: 8

  # 3. 远见与权衡 (RL Hyperparameters)
  # 含义：折扣因子 (Gamma)。代表 AI 有多在乎“未来”的奖励。
  # 0.99 = 很有远见；0.5 = 鼠目寸光。
  # 建议：机器人任务通常用 0.99
  discount_factor: 0.99

  # 含义：GAE 参数 (Lambda)。用于平衡方差和偏差。
  # 建议：0.95 是 RL 界的黄金标准，别动它
  lambda: 0.95

  # 4. 学习速度 (Optimization)
  # 含义：学习率 (Learning Rate)。步子迈多大。
  # 建议：3e-4 (0.0003) 是最常用的起始值
  learning_rate: 3.0e-04

  # 含义：学习率调度器。这在 Isaac Gym/Lab 里非常重要！
  # 作用：它会根据 KL 散度（策略变化的剧烈程度）自动调整学习率
  learning_rate_scheduler: KLAdaptiveLR

  # 含义：KL 散度的阈值。
  # 逻辑：如果策略变化超过 0.008，说明步子太大了，学习率会自动减半。
  # 建议：0.008 ~ 0.01 都是合理范围
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.008

  # 5. 数据预处理 (Preprocessing) —— ⚠️ 你的论文复现关键点！
  # 含义：状态归一化。把传感器数据（如高度 0.3, 速度 20.0）缩放到标准正态分布
  # 🔴 现状：null (关闭)。这对于四足机器人是 ❌ 错误的！
  # ✅ 修改：必须改成 RunningStandardScaler，否则很难收敛
  state_preprocessor: null
  state_preprocessor_kwargs: null

  # 含义：价值归一化。把 Critic 预测的分数也归一化
  # 建议：通常也开启 RunningStandardScaler
  value_preprocessor: null
  value_preprocessor_kwargs: null

  # 6. 训练稳定性 (Clipping & Safety)
  # 含义：随机探索步数。PPO 是 On-Policy 的，这里通常设为 0
  random_timesteps: 0
  learning_starts: 0

  # 含义：梯度裁剪。防止梯度爆炸（更新幅度过大导致数值溢出）
  grad_norm_clip: 1.0

  # 含义：PPO 核心参数。限制新旧策略的差异不能超过 20%
  # 建议：0.2 是标准值，别动
  ratio_clip: 0.2

  # 含义：限制 Critic 对价值预测的更新幅度
  value_clip: 0.2
  clip_predicted_values: True

  # 7. 奖励与探索 (Rewards)
  # 含义：熵奖励系数。鼓励 AI “多尝试不同动作”
  # 建议：AMP 任务通常设为 0.0 或极小值 (0.001)，因为我们希望动作像真狗一样稳定，不要瞎晃
  entropy_loss_scale: 0.0

  # 含义：价值损失的权重。Critic 训练的重要性
  value_loss_scale: 2.0

  # 含义：KL 散度目标。如果设了 scheduler，这个可以设为 0
  kl_threshold: 0.0

  # 含义：奖励缩放。把所有奖励乘以 1.0 (没变)
  # 有时候为了数值稳定，会缩放到 0.1 或 0.01
  rewards_shaper_scale: 1.0

  # 含义：超时处理。
  # True = 超时被视为“任务没完成但没死” (Bootstrap)
  # False = 超时被视为“任务结束”
  # 建议：对于无限时间任务（走路），通常设为 True 或 False 影响不大，Isaac Lab 默认 False
  time_limit_bootstrap: False

  # 8. 实验记录 (Logging)
  experiment:
    directory: "cartpole"       # 日志存哪？
    experiment_name: ""         # 实验名叫啥？
    write_interval: auto        # 多久写一次 Tensorboard
    checkpoint_interval: auto   # 多久存一次模型 (.pt 文件)
```

### 3.2.4 trainer

```yaml
trainer:
  class: SequentialTrainer
  timesteps: 2400 #意思是：“循环跑 2400 步就停下来”。
  environment_info: log
```

# 4 register
> source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/__init__.py
## 4.1


```python
gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", #环境 ManagerBasedRLEnvCfg/DirectRLEnv

    # 3. 关闭检查 (Disable Checker)
    # 必须设为 True。因为 Isaac Lab 是 GPU 并行环境，
    # 不符合传统 Gym 对 CPU 环境的严格格式检查。
    disable_env_checker=True,

    # 4. 核心参数 (kwargs)
    # 这里是传递给 "核心引擎" 的具体设置
    kwargs={
        # ====================================================
        # A. 环境配置 (物理世界)
        # ----------------------------------------------------
        # 指向你的环境配置类 (EnvCfg)。
        # 决定了：机器人长啥样、重力多少、观测什么数据、奖励怎么算。
        # 格式："{模块路径}:{类名}"
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg", #ManagerBasedRLEnvCfg/DirectRLEnvCfg

        # ====================================================
        # B. 算法配置 (大脑训练)
        # ----------------------------------------------------
        # 指向 RSL-RL 的训练参数配置类 (RunnerCfg)。
        # 决定了：学习率(lr)、批次大小(batch_size)、PPO 参数等。
        # 如果你只用 rsl_rl，写这一行就足够了！其他的 sb3, skrl 都可以删掉。
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",

        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerWithSymmetryCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
```