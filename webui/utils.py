__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import json
import locale
import platform
import yaml
import gradio as gr
import logging
from ml_collections import ConfigDict

from utils.constant import *
from utils.logger import get_logger, set_log_level
from tools.i18n import I18nAuto


# load and save config files
def load_configs(config_path):
	if config_path.endswith(".json"):
		with open(config_path, "r", encoding="utf-8") as f:
			return json.load(f)
	elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
		with open(config_path, "r", encoding="utf-8") as f:
			return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def save_configs(config, config_path):
	if config_path.endswith(".json"):
		with open(config_path, "w", encoding="utf-8") as f:
			json.dump(config, f, indent=4)
	elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
		with open(config_path, "w", encoding="utf-8") as f:
			yaml.dump(config.to_dict(), f)


def color_config(config):
	def format_dict(d):
		items = []
		for k, v in sorted(d.items()):
			colored_key = f"\033[0;33m{k}\033[0m"
			if isinstance(v, dict):
				formatted_value = f"{{{format_dict(v)}}}"
			else:
				formatted_value = str(v)
			items.append(f"{colored_key}: {formatted_value}")
		return ", ".join(items)

	return f"{{{format_dict(config)}}}"


# get language from config file and setup i18n, model download main link
def get_language():
	try:
		config = load_configs(WEBUI_CONFIG)
		language = config["settings"].get("language", "Auto")
	except:
		language = "Auto"

	if language == "Auto":
		language = locale.getdefaultlocale()[0]
	return language


def get_main_link():
	try:
		config = load_configs(WEBUI_CONFIG)
		main_link = config["settings"]["download_link"]
	except:
		main_link = "Auto"

	if main_link == "Auto":
		main_link = "hf-mirror.com" if get_language() == "zh_CN" else "huggingface.co"
	return main_link


logger = get_logger()
i18n = I18nAuto(get_language())


# webui restart function
def webui_restart():
	logger.info("Restarting WebUI...")
	os.execl(PYTHON, PYTHON, *sys.argv)


# setup webui debug mode
def log_level_debug(isdug):
	config = load_configs(WEBUI_CONFIG)
	if isdug:
		set_log_level(logger, logging.DEBUG)
		config["settings"]["debug"] = True
		save_configs(config, WEBUI_CONFIG)
		logger.info("Console log level set to \033[34mDEBUG\033[0m")
		return i18n("已开启调试日志")
	else:
		set_log_level(logger, logging.INFO)
		config["settings"]["debug"] = False
		save_configs(config, WEBUI_CONFIG)
		logger.info("Console log level set to \033[32mINFO\033[0m")
		return i18n("已关闭调试日志")


"""
following 5 functions are used for loading and getting model information.
- load_selected_model: return downloaded model list according to selected model type
- load_msst_model: return all downloaded msst model list
- get_msst_model: return model path, config path, model type, download link according to model name
- load_vr_model: return all downloaded uvr model list
- get_vr_model: return primary stem, secondary stem, model url, model path according to model name
"""


def load_selected_model(model_type=None):
	if not model_type:
		webui_config = load_configs(WEBUI_CONFIG)
		model_type = webui_config["inference"]["model_type"]
	if model_type:
		downloaded_model = []
		model_dir = os.path.join(MODEL_FOLDER, model_type)
		if not os.path.exists(model_dir):
			return None
		for files in os.listdir(model_dir):
			if files.endswith((".ckpt", ".th", ".chpt")):
				try:
					get_msst_model(files, model_type)
					downloaded_model.append(files)
				except:
					continue
		return downloaded_model
	return None


def load_msst_model():
	model_list = []
	model_classes = ["multi_stem_models", "single_stem_models", "vocal_models"]
	model_dir = [os.path.join(MODEL_FOLDER, keys) for keys in model_classes]
	for dirs in model_dir:
		for files in os.listdir(dirs):
			if files.endswith((".ckpt", ".th", ".chpt")):
				model_list.append(files)
	return model_list


def get_msst_model(model_name, model_type=None):
	# config = load_configs(MSST_MODEL)
	# main_link = get_main_link()
	# model_type = [model_type] if model_type else config.keys()

	# for keys in model_type:
	#     for model in config[keys]:
	#         if model["name"] == model_name:
	#             model_type = model["model_type"]
	#             model_path = os.path.join(MODEL_FOLDER, keys, model_name)
	#             config_path = model["config_path"]
	#             download_link = model["link"]
	#             try:
	#                 download_link = download_link.replace("huggingface.co", main_link)
	#             except:
	#                 pass
	#             return model_path, config_path, model_type, download_link

	# if os.path.isfile(os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json")):
	#     unofficial_config = load_configs(os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json"))
	#     for keys in model_type:
	#         for model in unofficial_config[keys]:
	#             if model["name"] == model_name:
	#                 model_type = model["model_type"]
	#                 model_path = os.path.join(MODEL_FOLDER, keys, model_name)
	#                 config_path = model["config_path"]
	#                 download_link = model["link"]
	#                 return model_path, config_path, model_type, download_link
	# raise gr.Error(i18n("模型不存在!"))

	config = load_configs(MODELS_INFO)
	main_link = get_main_link()
	model_type = [model_type] if model_type else ["multi_stem_models", "single_stem_models", "vocal_models"]
	if not model_name in config.keys():
		# print(model_name, config.keys())
		raise gr.Error(i18n("模型不存在!"))
	model = config[model_name]
	model_path = model["target_position"]
	config_path = model_path.replace("pretrain", "configs") + ".yaml"
	download_link = model["link"]
	model_type = model["model_type"]
	try:
		download_link = download_link.replace("huggingface.co", main_link)
	except:
		pass

	return model_path, config_path, model_type, download_link


def load_vr_model():
	downloaded_model = []
	config = load_configs(WEBUI_CONFIG)
	vr_model_path = config["settings"]["uvr_model_dir"]
	for files in os.listdir(vr_model_path):
		if files.endswith(".pth"):
			try:
				get_vr_model(files)
				downloaded_model.append(files)
			except:
				continue
	return downloaded_model


def get_vr_model(model):
	config = load_configs(MODELS_INFO)
	model_path = load_configs(WEBUI_CONFIG)["settings"]["uvr_model_dir"]
	main_link = get_main_link()

	for keys in config.keys():
		if keys == model:
			primary_stem = config[keys]["primary_stem"]
			secondary_stem = config[keys]["secondary_stem"]
			model_url = config[keys]["link"]
			try:
				model_url = model_url.replace("huggingface.co", main_link)
			except:
				pass
			return primary_stem, secondary_stem, model_url, model_path
	raise gr.Error(i18n("模型不存在!"))


# get model size and sha256 according to model name and model_info.json
def load_model_info(model_name):
	model_info = load_configs(MODELS_INFO)
	if model_name in model_info.keys():
		model_size = model_info[model_name].get("model_size", "Unknown")
		share256 = model_info[model_name].get("sha256", "Unknown")
		if model_size != "Unknown":
			model_size = round(int(model_size) / 1024 / 1024, 2)
	else:
		model_size = "Unknown"
		share256 = "Unknown"
	return model_size, share256


# update dropdown model list in webui according to selected model type
def update_model_name(model_type):
	if model_type == "UVR_VR_Models":
		model_map = load_vr_model()
		return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)
	else:
		model_map = load_selected_model(model_type)
		return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)


# change button visibility according to selected inference type
def change_to_audio_infer():
	return (gr.Button(i18n("输入音频分离"), variant="primary", visible=True), gr.Button(i18n("输入文件夹分离"), variant="primary", visible=False))


def change_to_folder_infer():
	return (gr.Button(i18n("输入音频分离"), variant="primary", visible=False), gr.Button(i18n("输入文件夹分离"), variant="primary", visible=True))


"""
following 4 functions are used for file and folder selection and open selected folder
- select_folder: use tkinter to select a folder and return the selected folder path
- select_yaml_file: use tkinter to select a yaml file and return the selected file path
- select_file: use tkinter to select a file and return the selected file path
- open_folder: open the selected folder in file explorer according to the selected folder path

**Must put import modules in functions to avoid import error when running on cloud plantform.**
"""


def select_folder():
	import tkinter as tk
	from tkinter import filedialog

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	selected_dir = filedialog.askdirectory()
	root.destroy()
	return selected_dir


def select_yaml_file():
	import tkinter as tk
	from tkinter import filedialog

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	selected_file = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
	root.destroy()
	return selected_file


def select_file():
	import tkinter as tk
	from tkinter import filedialog

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	selected_file = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
	root.destroy()
	return selected_file


def open_folder(folder):
	if folder == "":
		raise gr.Error(i18n("请先选择文件夹!"))
	os.makedirs(folder, exist_ok=True)
	absolute_path = os.path.abspath(folder)
	if platform.system() == "Windows":
		os.system(f"explorer {absolute_path}")
	elif platform.system() == "Darwin":
		os.system(f"open {absolute_path}")
	else:
		os.system(f"xdg-open {absolute_path}")


# error manager, add more detailed solutions according to the error message
def detailed_error(e):
	e = str(e)
	m = None

	if "CUDA out of memory" in e or "CUBLAS_STATUS_NOT_INITIALIZED" in e:
		m = i18n("显存不足, 请尝试减小batchsize值和chunksize值后重试。")
	elif "页面文件太小" in e or "DataLoader worker" in e or "DLL load failed while" in e or "[WinError 1455]" in e:
		m = i18n("内存不足，请尝试增大虚拟内存后重试。若分离时出现此报错，也可尝试将推理音频裁切短一些，分段分离。")
	elif "ffprobe not found" in e:
		m = i18n("FFmpeg未找到，请检查FFmpeg是否正确安装。若使用的是整合包，请重新安装。")
	elif "failed reading zip archive" in e:
		m = i18n("模型损坏，请重新下载并安装模型后重试。")
	elif "No such file or directory" in e or "系统找不到" in e or "[WinError 3]" in e or "[WinError 2]" in e or "The system cannot find the file specified" in e:
		m = i18n("文件或路径不存在，请根据错误指示检查是否存在该文件。")

	if m:
		e = m + "\n" + e
	return e
