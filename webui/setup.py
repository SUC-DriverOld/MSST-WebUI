__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import os
import shutil
from webui.utils import i18n, logger
from utils.constant import *
from webui.utils import load_configs, save_configs, log_level_debug, get_main_link


def copy_folders():
	if os.path.exists("configs"):
		shutil.rmtree("configs")
	shutil.copytree("configs_backup", "configs")
	if os.path.exists("data"):
		shutil.rmtree("data")
	shutil.copytree("data_backup", "data")


def update_configs_folder():
	config_dir = "configs"
	config_backup_dir = "configs_backup"

	for dirpath, _, files in os.walk(config_backup_dir):
		relative_path = os.path.relpath(dirpath, config_backup_dir)
		target_dir = os.path.join(config_dir, relative_path)
		if not os.path.exists(target_dir):
			os.makedirs(target_dir)

		for file in files:
			source_file = os.path.join(dirpath, file)
			target_file = os.path.join(target_dir, file)
			if not os.path.exists(target_file):
				shutil.copyfile(source_file, target_file)


def set_debug(args, iswebui=False):
	debug = False
	if iswebui and os.path.exists(WEBUI_CONFIG):
		debug = load_configs(WEBUI_CONFIG)["settings"].get("debug", False)
	if args.debug or debug:
		os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
		log_level_debug(True)
	else:
		import warnings

		warnings.filterwarnings("ignore")
		log_level_debug(False)


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
		except:
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

		if os.path.exists("data"):
			shutil.rmtree("data")
		shutil.copytree("data_backup", "data")

		update_configs_folder()
		logger.debug("Copied new configs from configs_backup to configs")

		save_configs(webui_config_backup, WEBUI_CONFIG)
		webui_config = webui_config_backup
		logger.debug("Merging old config with new config")

	if webui_config["settings"].get("auto_clean_cache", False):
		shutil.rmtree("cache")
		os.makedirs("cache", exist_ok=True)
		logger.info(i18n("成功清理Gradio缓存"))

	main_link = get_main_link()
	os.environ["HF_HOME"] = os.path.abspath(MODEL_FOLDER)
	os.environ["HF_ENDPOINT"] = "https://" + main_link
	ffmpeg_path = os.path.abspath("ffmpeg/bin/")
	os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
	os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("cache/")

	logger.debug("Set HF_HOME to: " + os.path.abspath(MODEL_FOLDER))
	logger.debug("Set HF_ENDPOINT to: " + "https://" + main_link)
	logger.debug("Set ffmpeg PATH to: " + os.path.abspath("ffmpeg/bin/"))
	logger.debug("Set GRADIO_TEMP_DIR to: " + os.path.abspath("cache/"))

	return webui_config
