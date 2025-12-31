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

# 2.define a env

>/home/oiioaa/Desktop/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py

## 2.1 dependencies

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

## 2.2 scene
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

## 2.3 action

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    #AI 输出的数字，要被当作【推力】(Effort)，施加在小车的【滑轨关节】上，并且要【放大100倍】

```

## 2.4 obs(Actor&Critic)

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

## 2.5 event(random reset)

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

## 2.6 reward

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

## 2.7 terminate

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

## 2.8 total

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

## 2.9 camera

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

### 2.9.1 add camera(RGB&)

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


### 2.9.2 obs

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

### 2.9.3 env

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

# 3 define Runner
> /home/oiioaa/Desktop/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/agents/rsl_rl_ppo_cfg.py

```python
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

import isaaclab_tasks.manager_based.classic.cartpole.mdp.symmetry as symmetry
```
## 3.1 Runner

```python
@configclass
class MyLSTMPolicyCfg(RslRlPpoActorCriticCfg):
    rnn_type: str = "lstm"
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    class_name = "ActorCriticRecurrent"

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

# 4 register
> source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/__init__.py
## 4.1
```python
gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",

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
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",

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