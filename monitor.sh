#!/bin/bash
# ==================== [ 配置区域 ] ====================
REMOTE_USER="oper"
REMOTE_IP="192.168.169.123"       # ROG 的 IP
REMOTE_PROJECT_DIR="/home/oper/Public/ZXZ/IsaacLab" # 远程项目路径
REMOTE_TENSORBOARD="/home/oper/miniconda3/envs/env_IsaacSim450/bin/tensorboard
"

# 这是一个 SSH 隧道命令：
# -L 6006:localhost:6006 意思是：把远程的 6006 端口，映射到我本地的 6006 端口
# -t 强制分配伪终端，方便你按 Ctrl+C 关闭
# source ~/.bashrc && conda activate rl2 : 确保环境激活
# tensorboard ... : 启动服务

echo "📊 正在连接远程 TensorBoard..."
echo "✅ 连接成功后，请在【本地浏览器】打开: http://localhost:6006"
echo "❌ 按 Ctrl+C 停止监控"

ssh -L 6006:localhost:6006 -t $REMOTE_USER@$REMOTE_IP \
    "cd $REMOTE_PROJECT_DIR; $REMOTE_TENSORBOARD --logdir logs/rsl_rl --port 6006 --bind_all"