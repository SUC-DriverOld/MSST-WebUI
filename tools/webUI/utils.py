import os
import json
import locale
import platform
import yaml
import rich
import tkinter as tk
import gradio as gr
import psutil
import subprocess
import threading
from tkinter import filedialog
from torch import cuda, backends
from ml_collections import ConfigDict

from tools.webUI.constant import *

stop_all_threads = False
stop_infer_flow = False

def get_stop_infer_flow():
    return stop_infer_flow

def change_stop_infer_flow():
    global stop_infer_flow
    stop_infer_flow = False

def webui_restart():
    os.execl(PYTHON, PYTHON, *sys.argv)

def i18n(key):
    try:
        config = load_configs(WEBUI_CONFIG)
        if config["settings"]["language"]== "Auto":
            language = locale.getdefaultlocale()[0]
        else: 
            language = config['settings']['language']
    except Exception:
        language = locale.getdefaultlocale()[0]

    if language == "zh_CN":
        return key
    if not os.path.exists(path=f"tools/i18n/locale/{language}.json"):
        language = "en_US"

    with open(file=f"tools/i18n/locale/{language}.json", mode="r", encoding="utf-8") as f:
        language_list = json.load(f)

    return language_list.get(key, key)

def get_device():
    try:
        if cuda.is_available():
            gpus = []
            n_gpu = cuda.device_count()
            for i in range(n_gpu):
                gpus.append(f"{i}: {cuda.get_device_name(i)}")
            return gpus
        elif backends.mps.is_available():
            return [i18n("使用MPS")]
        else:
            return [i18n("无可用的加速设备, 使用CPU")]
    except Exception:
        return [i18n("设备检查失败")]

def get_platform():
    os_name = platform.system()
    os_version = platform.version()
    machine = platform.machine()
    return f"System: {os_name}, Version: {os_version}, Machine: {machine}"

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

def print_command(command, title="Use command"):
    console = rich.console.Console()
    panel = rich.panel.Panel(command, title=title, style=rich.style.Style(color="green"), border_style="green")
    console.print(panel)

def load_augmentations_config():
    try:
        with open("configs/augmentations_template.yaml", 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return i18n("错误: 无法找到增强配置文件模板, 请检查文件configs/augmentations_template.yaml是否存在。")

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
                    get_msst_model(files)
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

def get_msst_model(model_name):
    config = load_configs(MSST_MODEL)
    webui_config = load_configs(WEBUI_CONFIG)
    main_link = webui_config['settings']['download_link']

    if main_link == "Auto":
        language = locale.getdefaultlocale()[0]
        if language in ["zh_CN", "zh_TW", "zh_HK", "zh_SG"]:
            main_link = "hf-mirror.com"
        else: 
            main_link = "huggingface.co"

    for keys in config.keys():
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

    try:
        unofficial_config = load_configs(os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json"))
    except FileNotFoundError:
        raise gr.Error(i18n("模型不存在!"))

    for keys in unofficial_config.keys():
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
    webui_config = load_configs(WEBUI_CONFIG)
    model_path = webui_config['settings']['uvr_model_dir']
    main_link = webui_config['settings']['download_link']

    if main_link == "Auto":
        language = locale.getdefaultlocale()[0]
        if language in ["zh_CN", "zh_TW", "zh_HK", "zh_SG"]:
            main_link = "hf-mirror.com"
        else: 
            main_link = "huggingface.co"

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

    try:
        unofficial_config = load_configs(os.path.join(UNOFFICIAL_MODEL, "unofficial_vr_model.json"))
    except FileNotFoundError:
        raise gr.Error(i18n("模型不存在!"))

    for keys in unofficial_config.keys():
        if keys == model:
            primary_stem = unofficial_config[keys]["primary_stem"]
            secondary_stem = unofficial_config[keys]["secondary_stem"]
            model_url = unofficial_config[keys]["download_link"]
            return primary_stem, secondary_stem, model_url, model_path

    raise gr.Error(i18n("模型不存在!"))

def load_vr_model_stem(model):
    primary_stem, secondary_stem, _, _= get_vr_model(model)
    return gr.Checkbox(label=f"{primary_stem} Only", value=False, interactive=True), gr.Checkbox(label=f"{secondary_stem} Only", value=False, interactive=True)

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

def run_command(command):
    global stop_all_threads
    stop_all_threads = False

    print_command(command)
    try:
        process = subprocess.Popen(command, shell=True)
        proc = psutil.Process(process.pid)

        while process.poll() is None:
            if stop_all_threads:
                for child in proc.children(recursive=True):
                    child.terminate()
                process.terminate()
                stop_all_threads = False
                return

        if process.returncode != 0:
            gr.Error(i18n("发生错误! 请前往终端查看详细信息"))

    except Exception as e:
        print(e)
        raise gr.Error(i18n("发生错误! 请前往终端查看详细信息"))

def stop_all_thread():
    global stop_all_threads, stop_infer_flow

    for thread in threading.enumerate():
        if thread.name in ["msst_inference", "vr_inference", "msst_training", "msst_valid"]:
            stop_all_threads = True
            stop_infer_flow = True
            gr.Info(i18n("已停止进程"))
            print("[INFO] " + i18n("已停止进程"))