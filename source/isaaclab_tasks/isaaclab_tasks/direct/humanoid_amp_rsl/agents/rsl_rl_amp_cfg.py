from isaaclab.utils import configclass
# 导入上面定义的类
from isaaclab_rl.rsl_rl import (
    RslRlAmpOnPolicyRunnerCfg, 
    RslRlAmpPpoAlgorithmCfg, 
    RslRlPpoActorCriticRecurrentCfg,
    RslRlAmpDiscriminatorCfg,
    RslRlAmpLoaderCfg
)

@configclass
class HumanoidAmpPPORunnerCfg(RslRlAmpOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "humanoid_amp_example"
    run_name = "lstm_amp"

    # 1. 策略网络配置 (支持 LSTM)
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256],
        critic_hidden_dims=[512, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

    # 2. AMP 算法配置
    algorithm = RslRlAmpPpoAlgorithmCfg(
        value_loss_coef=1.0, 
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=1,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # AMP 特有
        amp_replay_buffer_size=200000
    )

    # 3. 判别器配置
    discriminator = RslRlAmpDiscriminatorCfg(
        hidden_dims=[1024, 512],
        reward_scale=2.0, # 风格奖励权重
        loss_type="BCEWithLogits",
        empirical_normalization=False
    )

    # 4. 专家数据集配置
    dataset = RslRlAmpLoaderCfg(
        amp_data_path="source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp_rsl/motions", # 需要修改为实际路径
        datasets={
            "humanoid_dance.npz": 1.0, # 文件名 : 权重
            # "humanoid_walk.npy": 0.5   # 支持混合多个动作文件
        },
        slow_down_factor=1.0
    )