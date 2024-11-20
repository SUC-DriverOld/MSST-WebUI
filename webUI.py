"""
Web UI Setup and Launch Script
Author: Sucial: https://github.com/SUC-DriverOld

This script is responsible for setting up and launching the Web User Interface (WebUI).
It manages the configuration and data directories, ensuring that the latest settings are
loaded, and initializes the necessary environment for the application.

Key Configurations:
- WEBUI_CONFIG: Path to the main WebUI configuration file.
- WEBUI_CONFIG_BACKUP: Backup path for the WebUI configuration.
- PRESETS: Path for storing preset configurations.
- MODEL_FOLDER: Directory for pre-trained models.
- THEME_FOLDER: Path for theme resources.
- PACKAGE_VERSION: Current version of the WebUI package.
- HF_HOME: Path for Hugging Face model cache.
- GRADIO_TEMP_DIR: Temporary directory for Gradio caching.

Environment Setup:
- Automatically creates required directories (input, results, cache).
- Copies backup configurations if the current ones are missing or outdated.
- Sets up environment variables for model management and tool paths (e.g., FFmpeg).

Device Management:
- Detects available hardware accelerators (CUDA, MPS, CPU) for optimized performance.
- Logs device information and issues warnings if no accelerators are found.

Command Line Arguments:
- --server_name: Specify the server IP address (default: Auto).
- --server_port: Specify the server port (default: Auto).
- --share: Enable sharing the WebUI (default: False).
- --debug: Enable debug mode (default: False).

Usage:
Run this script to initialize the WebUI and launch it in the browser. Ensure that
all necessary backups and dependencies are in place for successful execution.
"""

import os
import sys
import shutil
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from utils.constant import *


def copy_folders():
    if os.path.exists("configs"):
        shutil.rmtree("configs")
    shutil.copytree("configs_backup", "configs")
    if os.path.exists("data"):
        shutil.rmtree("data")
    shutil.copytree("data_backup", "data")


def setup_webui():
    logger.debug("Starting WebUI setup")

    os.makedirs("input", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    webui_config = load_configs(WEBUI_CONFIG)
    logger.debug(f"Loading WebUI config: {webui_config}")
    version = webui_config.get("version", None)

    if not version:
        try: 
            version = load_configs("data/version.json")["version"]
        except Exception: 
            copy_folders()
            version = PACKAGE_VERSION
            logger.warning("Can't find old version, copying backup folders")

    if version != PACKAGE_VERSION:
        logger.info(i18n("检测到") + version + i18n("旧版配置, 正在更新至最新版") + PACKAGE_VERSION)
        webui_config_backup = load_configs(WEBUI_CONFIG_BACKUP)

        for module in ["training", "inference", "tools", "settings"]:
            for key in webui_config_backup[module].keys():
                try: 
                    webui_config_backup[module][key] = webui_config[module][key]
                except KeyError: 
                    continue

        copy_folders()
        save_configs(webui_config_backup, WEBUI_CONFIG)
        logger.debug("Merging old config with new config")
        time.sleep(1)

    if webui_config["settings"].get("auto_clean_cache", False):
        shutil.rmtree("cache")
        os.makedirs("cache", exist_ok=True)
        logger.info(i18n("成功清理Gradio缓存"))

    main_link = get_main_link()
    os.environ["HF_HOME"] = os.path.abspath(MODEL_FOLDER)
    os.environ["HF_ENDPOINT"] = "https://" + main_link
    os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg/bin/")
    os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("cache/")

    logger.debug("Set HF_HOME to: " + os.path.abspath(MODEL_FOLDER))
    logger.debug("Set HF_ENDPOINT to: " + "https://" + main_link)
    logger.debug("Set ffmpeg PATH to: " + os.path.abspath("ffmpeg/bin/"))
    logger.debug("Set GRADIO_TEMP_DIR to: " + os.path.abspath("cache/"))


def set_debug(args):
    debug = False
    if os.path.isfile(WEBUI_CONFIG):
        webui_config = load_configs(WEBUI_CONFIG)
        debug = webui_config["settings"].get("debug", False)
    if args.debug or debug:
        log_level_debug(True)
    else:
        import warnings
        log_level_debug(False)
        warnings.filterwarnings("ignore")


if __name__ == "__main__":
    if not os.path.exists("data") or not os.path.exists("configs"):
        copy_folders()

    import argparse
    from webui.utils import i18n, logger

    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
    parser.add_argument("--server_name", type=str, default=None, help="Server IP address (Default: Auto).")
    parser.add_argument("--server_port", type=int, default=None, help="Server port (Default: Auto).")
    parser.add_argument("--share", action="store_true", help="Enable share link (Default: False).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (Default: False).")
    args = parser.parse_args()

    logger.info(i18n("正在启动WebUI, 请稍等..."))
    logger.info(i18n("若启动失败, 请尝试以管理员身份运行此程序"))
    logger.warning(i18n("WebUI运行过程中请勿关闭此窗口!"))

    import app
    import platform
    from torch import cuda, backends
    from webui.utils import load_configs, save_configs, log_level_debug, get_main_link
    import multiprocessing
    multiprocessing.set_start_method('spawn')

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

    set_debug(args)
    setup_webui()
    webui_config = load_configs(WEBUI_CONFIG)

    if args.server_name:
        server_name = args.server_name
    else:
        server_name = "0.0.0.0" if webui_config["settings"].get("local_link", False) else None
    if args.server_port:
        server_port = args.server_port
    else:
        server_port = None if webui_config["settings"].get("port", 0) == 0 else webui_config["settings"].get("port", 0)
    if args.share or webui_config["settings"].get("share_link", False):
        share = True
    else:
        share = False
    theme_path = os.path.join(THEME_FOLDER, webui_config["settings"]["theme"])

    logger.debug(f"Launching WebUI with parameters: server_name={server_name}, server_port={server_port}, share={share}")

    app.app(
        platform=platform_info, device=devices, force_cpu=force_cpu, theme=theme_path
    ).launch(
        inbrowser=True, share=share, server_name=server_name, server_port=server_port
    )
