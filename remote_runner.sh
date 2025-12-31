#!/bin/bash

# ==================== [ 配置区域 ] ====================

# 1. 远程服务器信息 (ROG 笔记本)
REMOTE_USER="oper"
REMOTE_IP="192.168.169.123"       # 换成的真实局域网 IP
REMOTE_PROJECT_DIR="/home/oper/Public/ZXZ/IsaacLab" # 服务器上 Isaac Lab 的位置
REMOTE_PYTHON="/home/oper/miniconda3/envs/env_IsaacSim450/bin/python" # 服务器上 conda 环境的 python 路径

# 2. 本地路径
LOCAL_PROJECT_DIR="./"          # 就在当前目录
LOCAL_LOGS_DIR="./logs"         # 本地存放 Log 的位置

# 3. 任务名称 (修改这里来切换任务)
TASK_NAME="Isaac-Velocity-Flat-Anymal-C-v0"
# TASK_NAME="***********"

# 4. 训练参数
HEADLESS=true                   # 训练时是否不显示窗口 (通常设为 true 跑得快)
MAX_ITERATIONS=100              # 训练多少轮
NUM_ENVS=4096                 # 远程训练时使用多少并行环境

# =====================================================

echo "🚀 [Step 1] 同步代码: 本地 -> 远程..."
# 使用 rsync 增量同步代码
# --exclude: 排除不需要上传的大文件、git文件、本地日志
rsync -avz --delete \
    --exclude '.git/' \
    --exclude '.gitignore' \
    --exclude '__pycache__/' \
    --exclude 'logs/' \
    --exclude 'videos/' \
    --exclude '*.egg-info' \
    --exclude '.vscode' \
    --exclude 'remote_runner.sh' \
    --exclude 'docs' \
    --exclude 'monitor.sh' \
    $LOCAL_PROJECT_DIR $REMOTE_USER@$REMOTE_IP:$REMOTE_PROJECT_DIR

if [ $? -ne 0 ]; then
    echo "❌ 代码同步失败，请检查网络或 SSH 配置。"
    exit 1
fi
echo "✅ 代码同步完成。"

echo "🏃 [Step 2] 开始远程训练..."
# 拼接远程命令
# source ~/.bashrc 确保 conda 能用
# cd 到目录 -> 运行 train.py
SSH_CMD="cd $REMOTE_PROJECT_DIR; \
         $REMOTE_PYTHON scripts/reinforcement_learning/rsl_rl/train.py \
         --task $TASK_NAME \
         --max_iterations $MAX_ITERATIONS \
         --num_envs $NUM_ENVS"\
         

if [ "$HEADLESS" = true ]; then
    SSH_CMD="$SSH_CMD --headless"
fi

# 执行远程命令
ssh $REMOTE_USER@$REMOTE_IP "$SSH_CMD"

if [ $? -ne 0 ]; then
    echo "❌ 训练异常中断。"
    exit 1
fi
echo "✅ 远程训练结束。"

echo "📥 [Step 3] 同步 Logs: 远程 -> 本地..."
# 只拉取 logs 文件夹，注意 logs/ 后面的斜杠很重要
rsync -avz $REMOTE_USER@$REMOTE_IP:$REMOTE_PROJECT_DIR/logs/ $LOCAL_LOGS_DIR/

echo "✅ Logs 已下载到本地 $LOCAL_LOGS_DIR"

# echo "🎮 [Step 4] 尝试在本地运行 Play (可视化)..."
# # 自动寻找最新的实验文件夹（简单的查找逻辑）
# # 假设你的 logs 结构是 logs/rsl_rl/任务名/日期/...
# LATEST_EXP_DIR=$(ls -td logs/rsl_rl/*/ | head -1)

# if [ -z "$LATEST_EXP_DIR" ]; then
#     echo "⚠️ 没找到本地 Log 文件夹，无法播放。"
# else
#     echo "正在播放最新的实验: $LATEST_EXP_DIR"
#     # 假设本地也有环境，直接调用本地的 python
#     # 注意：这里需要你本地也装好了 Isaac Lab
#     python scripts/reinforcement_learning/rsl_rl/play.py \
#         --task $TASK_NAME \
#         --num_envs 1 \
#         --load_run $(basename $LATEST_EXP_DIR) 
# fi