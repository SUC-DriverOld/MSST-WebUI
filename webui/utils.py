import json
import locale
import platform
import yaml
import tkinter as tk
import gradio as gr
import logging
from tkinter import filedialog
from ml_collections import ConfigDict

from utils.constant import *
from utils.logger import get_logger, set_log_level
from tools.i18n import i18n


def load_configs(config_path):
    if config_path.endswith('.json'):
        with open(config_path, 'r', encoding="utf-8") as f:
            return json.load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

def save_configs(config, config_path):
    if config_path.endswith('.json'):
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config, f, indent=4)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f)

def get_language():
    config = load_configs(WEBUI_CONFIG)
    language = config['settings']['language']
    if language == "Auto":
        language = locale.getdefaultlocale()[0]
    return language


logger = get_logger()
i18n = i18n.I18nAuto(get_language())


def webui_restart():
    logger.info("Restarting WebUI...")
    os.execl(PYTHON, PYTHON, *sys.argv)

def get_main_link():
    config = load_configs(WEBUI_CONFIG)
    main_link = config['settings']['download_link']
    if main_link == "Auto":
        language = get_language()
        main_link = "hf-mirror.com" if language == "zh_CN" else "huggingface.co"
    return main_link

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

def load_selected_model(model_type=None):
    if not model_type:
        webui_config = load_configs(WEBUI_CONFIG)
        model_type = webui_config["inference"]["model_type"]
    if model_type:
        downloaded_model = []
        model_dir = os.path.join(MODEL_FOLDER, model_type)

        for files in os.listdir(model_dir):
            if files.endswith(('.ckpt', '.th', '.chpt')):
                try: 
                    get_msst_model(files, model_type)
                    downloaded_model.append(files)
                except: 
                    continue
        return downloaded_model
    return None

def load_msst_model():
    config = load_configs(MSST_MODEL)
    model_list = []
    model_dir = [os.path.join(MODEL_FOLDER, keys) for keys in config.keys()]
    for dirs in model_dir:
        for files in os.listdir(dirs):
            if files.endswith(('.ckpt', '.th', '.chpt')):
                model_list.append(files)
    return model_list

def get_msst_model(model_name, model_type=None):
    config = load_configs(MSST_MODEL)
    main_link = get_main_link()
    model_type = [model_type] if model_type else config.keys()

    for keys in model_type:
        for model in config[keys]:
            if model["name"] == model_name:
                model_type = model["model_type"]
                model_path = os.path.join(MODEL_FOLDER, keys, model_name)
                config_path = model["config_path"]
                download_link = model["link"]
                try:
                    download_link = download_link.replace("huggingface.co", main_link)
                except:
                    pass
                return model_path, config_path, model_type, download_link

    if os.path.isfile(os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json")):
        unofficial_config = load_configs(os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json"))
        for keys in model_type:
            for model in unofficial_config[keys]:
                if model["name"] == model_name:
                    model_type = model["model_type"]
                    model_path = os.path.join(MODEL_FOLDER, keys, model_name)
                    config_path = model["config_path"]
                    download_link = model["link"]
                    return model_path, config_path, model_type, download_link
    raise gr.Error(i18n("模型不存在!"))

def load_vr_model():
    downloaded_model = []
    config = load_configs(WEBUI_CONFIG)
    vr_model_path = config['settings']['uvr_model_dir']
    for files in os.listdir(vr_model_path):
        if files.endswith('.pth'):
            try: 
                get_vr_model(files)
                downloaded_model.append(files)
            except: 
                continue
    return downloaded_model

def get_vr_model(model):
    config = load_configs(VR_MODEL)
    model_path = load_configs(WEBUI_CONFIG)['settings']['uvr_model_dir']
    main_link = get_main_link()

    for keys in config.keys():
        if keys == model:
            primary_stem = config[keys]["primary_stem"]
            secondary_stem = config[keys]["secondary_stem"]
            model_url = config[keys]["download_link"]
            try:
                model_url = model_url.replace("huggingface.co", main_link)
            except: 
                pass
            return primary_stem, secondary_stem, model_url, model_path

    if os.path.isfile(os.path.join(UNOFFICIAL_MODEL, "unofficial_vr_model.json")):
        unofficial_config = load_configs(os.path.join(UNOFFICIAL_MODEL, "unofficial_vr_model.json"))
        for keys in unofficial_config.keys():
            if keys == model:
                primary_stem = unofficial_config[keys]["primary_stem"]
                secondary_stem = unofficial_config[keys]["secondary_stem"]
                model_url = unofficial_config[keys]["download_link"]
                return primary_stem, secondary_stem, model_url, model_path
    raise gr.Error(i18n("模型不存在!"))


def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_dir = filedialog.askdirectory()
    root.destroy()
    return selected_dir

def select_yaml_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_file = filedialog.askopenfilename(
        filetypes=[('YAML files', '*.yaml')])
    root.destroy()
    return selected_file

def select_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_file = filedialog.askopenfilename(
        filetypes=[('All files', '*.*')])
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