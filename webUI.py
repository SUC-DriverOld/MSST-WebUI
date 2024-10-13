import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import shutil
import locale
import app
from torch import cuda, backends
from utils.constant import *
from tools.webUI.utils import i18n, load_configs, save_configs, logger

import warnings
warnings.filterwarnings("ignore")

import platform
platform_info = f"System: {platform.system()}, Version: {platform.version()}, Machine: {platform.machine()}"

gpus = []
if cuda.is_available():
    n_gpu = cuda.device_count()
    for i in range(n_gpu):
        gpus.append(f"{i}: {cuda.get_device_name(i)}")
elif backends.mps.is_available():
    gpus = [i18n("使用MPS")]
else:
    gpus = [i18n("无可用的加速设备, 使用CPU")]

def copy_folders():
    if os.path.exists("configs"):
        shutil.rmtree("configs")
    shutil.copytree("configs_backup", "configs")

    if os.path.exists("data"):
        shutil.rmtree("data")
    shutil.copytree("data_backup", "data")

def setup_webui(logger):
    logger.debug("Starting WebUI setup")
    logger.info("WebUI Version: " + PACKAGE_VERSION)
    logger.info(platform_info)

    if not os.path.exists("data") or not os.path.exists("configs"):
        copy_folders()
        logger.debug("Copying backup folders")

    os.makedirs("input", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    webui_config = load_configs(WEBUI_CONFIG)
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
        presets_config = load_configs(PRESETS)
        webui_config_backup = load_configs(WEBUI_CONFIG_BACKUP)

        for module in ["training", "inference", "tools", "settings"]:
            for key in webui_config_backup[module].keys():
                try: 
                    webui_config_backup[module][key] = webui_config[module][key]
                except KeyError: 
                    continue

        copy_folders()
        save_configs(webui_config_backup, WEBUI_CONFIG)
        save_configs(presets_config, PRESETS)

        logger.debug("Merging old config with new config")

    if webui_config["settings"].get("auto_clean_cache", False):
        shutil.rmtree("cache")
        os.makedirs("cache", exist_ok=True)
        logger.info(i18n("成功清理Gradio缓存"))

    main_link = webui_config['settings']['download_link']
    if main_link == "Auto":
        language = locale.getdefaultlocale()[0]
        if language in ["zh_CN", "zh_TW", "zh_HK", "zh_SG"]:
            main_link = "hf-mirror.com"
        else: 
            main_link = "huggingface.co"

    os.environ["HF_HOME"] = os.path.abspath(MODEL_FOLDER)
    os.environ["HF_ENDPOINT"] = "https://" + main_link
    os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg/bin/")
    os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("cache/")

    logger.debug("Set HF_HOME to " + os.path.abspath(MODEL_FOLDER))
    logger.debug("Set HF_ENDPOINT to " + "https://" + main_link)
    logger.debug("Set ffmpeg PATH to " + os.path.abspath("ffmpeg/bin/"))
    logger.debug("Set GRADIO_TEMP_DIR to " + os.path.abspath("cache/"))
    logger.info(i18n("设备信息: " + (gpus if len(gpus) > 1 else gpus[0])))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
    parser.add_argument("-sn", "--server_name", type=str, default=None)
    parser.add_argument("-sp", "--server_port", type=int, default=None)
    parser.add_argument("-share", action="store_true")
    args = parser.parse_args()

    setup_webui(logger)
    webui_config = load_configs(WEBUI_CONFIG)

    if args.server_name:
        server_name = args.server_name
    else:
        server_name = "0.0.0.0" if webui_config["settings"].get("local_link", False) else None

    if args.server_port:
        server_port = args.server_port
    else:
        server_port = None if webui_config["settings"].get("port", 0) == 0 else webui_config["settings"].get("port", 0)

    if args.share:
        share = True
    else:
        share = webui_config["settings"].get("share_link", False)

    logger.debug("WebUI parameters: " + str(args))
    logger.debug("Launching WebUI...")

    app.app(platform=platform_info, device=gpus).launch(inbrowser=True, share=share, server_name=server_name, server_port=server_port)
