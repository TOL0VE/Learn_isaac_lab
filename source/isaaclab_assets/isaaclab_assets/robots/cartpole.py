# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
