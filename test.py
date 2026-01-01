import argparse
# 1. 首先导入 AppLauncher
from isaaclab.app import AppLauncher

# 创建参数解析器 (AppLauncher 需要读取 --headless 等参数)
parser = argparse.ArgumentParser(description="Test Script")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 2. 【关键】启动仿真器
# 这一步会初始化 Isaac Sim，把 pxr, omni 等库加载进内存
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 3. 只有在 App 启动后，才能导入依赖 Isaac Sim 的模块！
# ---------------------------------------------------------
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

print("-" * 50)
print(f"✅ Success! Nucleus URL: {ISAACLAB_NUCLEUS_DIR}")
print("-" * 50)

# 4. 关闭仿真器
simulation_app.close()