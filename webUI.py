__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

"""
This file is responsible for initializing and launching the WebUI for the project. It performs multiple setup tasks,
including configuring system paths, setting up multiprocessing, verifying the existence of necessary directories,
and parsing command-line arguments for custom configurations. 

Key functionality includes:
- Modifying the system path to ensure proper module imports.
- Setting the multiprocessing start method to 'spawn' for cross-platform compatibility.
- Ensuring critical folders (e.g., "configs" and "data") exist by copying them from backup directories if necessary.
- Allowing users to configure WebUI launch parameters such as server IP, port, debugging options, and share link
  via command-line arguments.
- Detecting available hardware devices (GPU, MPS, or CPU) and logging the system's configuration for better transparency.
- Logging important startup messages and warnings regarding the environment setup and WebUI usage.
- Launching the WebUI with appropriate settings based on the user's input or default configuration.

Note:
- This source file contains critical setup and launch code for the WebUI and should not be modified unless necessary.
- Changes to the code could affect the launch behavior and system setup, leading to potential issues during execution.

This file ensures a smooth and customizable launch process for the WebUI, setting up all necessary components and
environments for the application to run efficiently.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"


def main(args):
	from webui.utils import i18n, logger

	logger.info(i18n("正在启动WebUI, 请稍等..."))
	logger.info(i18n("若启动失败, 请尝试以管理员身份运行此程序"))
	logger.warning(i18n("WebUI运行过程中请勿关闭此窗口!"))

	# import required modules
	import platform
	from webui.setup import set_debug, setup_webui
	from utils.constant import THEME_FOLDER, PACKAGE_VERSION
	from torch import cuda, backends

	# check available devices and force CPU if no GPU or MPS is available
	devices = {}
	force_cpu = False
	if cuda.is_available():
		for i in range(cuda.device_count()):
			devices[f"cuda{i}"] = f"{i}: {cuda.get_device_name(i)}"
		logger.info(i18n("检测到CUDA, 设备信息: ") + str(devices))
	elif backends.mps.is_available():
		devices = {"mps": i18n("使用MPS")}
		logger.info(i18n("检测到MPS, 使用MPS"))
	else:
		devices = {"cpu": i18n("无可用的加速设备, 使用CPU")}
		logger.warning(i18n("\033[33m未检测到可用的加速设备, 使用CPU\033[0m"))
		logger.warning(i18n("\033[33m如果你使用的是NVIDIA显卡, 请更新显卡驱动至最新版后重试\033[0m"))
		force_cpu = True
	platform_info = f"System: {platform.system()}, Machine: {platform.machine()}"
	logger.info(f"WebUI Version: {PACKAGE_VERSION}, {platform_info}")

	# set debug mode and setup WebUI
	set_debug(args, iswebui=True)
	webui_config = setup_webui()

	# setup WebUI launch parameters
	if args.ip_address:
		server_name = args.ip_address
	else:
		server_name = "0.0.0.0" if webui_config["settings"].get("local_link", False) else None
	if args.port:
		server_port = args.port
	else:
		server_port = None if webui_config["settings"].get("port", 0) == 0 else webui_config["settings"].get("port", None)
	if args.share or webui_config["settings"].get("share_link", False):
		share = True
	else:
		share = False
	theme_path = os.path.join(THEME_FOLDER, webui_config["settings"].get("theme", "theme_blue.json"))
	logger.debug(f"Launching WebUI with parameters: ip_address={server_name}, port={server_port}, share={share}")

	# launch WebUI
	from webui import app

	app.app(platform=platform_info, device=devices, force_cpu=force_cpu, theme=theme_path).queue().launch(
		inbrowser=True, share=share, server_name=server_name, server_port=server_port, show_api=False, favicon_path="docs/logo.png"
	)


if __name__ == "__main__":
	import multiprocessing
	import argparse
	import shutil
	import json
	from utils.constant import WEBUI_CONFIG

	multiprocessing.set_start_method("spawn", force=True)

	parser = argparse.ArgumentParser(description="WebUI for Music Source Separation Training", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
	parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
	parser.add_argument("-i", "--ip_address", type=str, default=None, help="Server IP address (Default: Auto).")
	parser.add_argument("-p", "--port", type=int, default=None, help="Server port (Default: Auto).")
	parser.add_argument("-s", "--share", action="store_true", help="Enable share link.")
	parser.add_argument("--use_cloud", action="store_true", help="Use special WebUI in cloud platforms.")
	parser.add_argument("--language", type=str, default=None, choices=[None, "Auto", "zh_CN", "zh_TW", "en_US", "ja_JP", "ko_KR"], help="Set WebUI language (Default: Auto).")
	parser.add_argument("--model_download_link", type=str, default=None, choices=[None, "Auto", "huggingface.co", "hf-mirror.com"], help="Set model download link (Default: Auto).")
	parser.add_argument("--factory_reset", action="store_true", help="Reset WebUI settings and model seetings to default, clear cache and exit.")
	args = parser.parse_args()

	if args.factory_reset:
		for dir in ["configs", "data", "cache", "tmpdir"]:
			if os.path.exists(dir):
				shutil.rmtree(dir)
		print("Factory reset completed.")
		os._exit(0)

	if not os.path.exists("configs"):
		shutil.copytree("configs_backup", "configs")
	if not os.path.exists("data"):
		shutil.copytree("data_backup", "data")

	if os.path.exists(WEBUI_CONFIG) and (args.language or args.model_download_link or args.debug):
		with open(WEBUI_CONFIG, "r", encoding="utf-8") as f:
			config = json.load(f)
		if args.language:
			config["settings"]["language"] = args.language
		if args.model_download_link:
			config["settings"]["download_link"] = args.model_download_link
		if args.use_cloud and args.debug:
			config["settings"]["debug"] = args.debug
		with open(WEBUI_CONFIG, "w", encoding="utf-8") as f:
			json.dump(config, f, indent=4)

	if args.use_cloud:  # if user uses webui in cloud platforms
		from tools.webUI_for_clouds.webUI_for_clouds import launch

		launch(server_name=args.ip_address, server_port=args.port, share=args.share)
	else:  # user uses webui in local environment
		main(args)
