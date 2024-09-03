import gradio as gr
import subprocess
import os
import sys
import re
import time
import shutil
import json
import requests
import yaml
import tkinter as tk
import librosa
import numpy as np
import pandas as pd
import webbrowser
import platform
import warnings
import locale
import threading
import psutil
from datetime import datetime
from ml_collections import ConfigDict
from tqdm import tqdm
from mir_eval.separation import bss_eval_sources
from tkinter import filedialog
from pydub import AudioSegment
from rich.console import Console
from torch import cuda, backends
from multiprocessing import cpu_count

PACKAGE_VERSION = "1.5.1"
WEBUI_CONFIG = "data/webui_config.json"
WEBUI_CONFIG_BACKUP = "data_backup/webui_config.json"
PRESETS = "data/preset_data.json"
MSST_MODEL = "data/msst_model_map.json"
VR_MODEL = "data/vr_model_map.json"
BACKUP = "backup/"
MODEL_FOLDER = "pretrain/"
TEMP_PATH = "temp"
UNOFFICIAL_MODEL = "config_unofficial"
MODEL_TYPE = ['bs_roformer', 'mel_band_roformer', 'segm_models', 'htdemucs', 'mdx23c', 'swin_upernet', 'bandit']
MODEL_CHOICES = ["vocal_models", "multi_stem_models", "single_stem_models", "UVR_VR_Models"]
FFMPEG = ".\\ffmpeg\\bin\\ffmpeg.exe" if os.path.isfile(".\\ffmpeg\\bin\\ffmpeg.exe") else "ffmpeg"
PYTHON = ".\\workenv\\python.exe" if os.path.isfile(".\\workenv\\python.exe") else sys.executable

language_dict = {
    "Auto": "Auto",
    "简体中文": "zh_CN",
    "繁體中文": "zh_TW",
    "English": "en_US",
    "日本語": "ja_JP",
    "😊": "emoji"
    }

warnings.filterwarnings("ignore")
stop_all_threads = False
stop_infer_flow = False

def setup_webui():
    def copy_folders():
        if os.path.exists("configs"):
            shutil.rmtree("configs")
        shutil.copytree("configs_backup", "configs")
        if os.path.exists("data"):
            shutil.rmtree("data")
        shutil.copytree("data_backup", "data")

    if not os.path.exists("data"):
        shutil.copytree("data_backup", "data")
        print(i18n("[INFO] 正在初始化data目录"))
    if not os.path.exists("configs"):
        shutil.copytree("configs_backup", "configs")
        print(i18n("[INFO] 正在初始化configs目录"))
    if not os.path.exists("input"): os.makedirs("input")
    if not os.path.exists("results"): os.makedirs("results")
    if os.path.exists("data"):
        webui_config = load_configs(WEBUI_CONFIG)
        version = webui_config.get("version", None)
        if not version:
            try: version = load_configs("data/version.json")["version"]
            except Exception: 
                copy_folders()
                version = PACKAGE_VERSION
        if version != PACKAGE_VERSION:
            print(i18n("[INFO] 检测到") + version + i18n("旧版配置, 正在更新至最新版") + PACKAGE_VERSION)
            presets_config = load_configs(PRESETS)
            webui_config_backup = load_configs(WEBUI_CONFIG_BACKUP)
            for module in ["training", "inference", "tools", "settings"]:
                for key in webui_config_backup[module]:
                    try: webui_config_backup[module][key] = webui_config[module][key]
                    except KeyError: continue
            copy_folders()
            save_configs(webui_config_backup, WEBUI_CONFIG)
            save_configs(presets_config, PRESETS)

    os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg/bin/")
    print(i18n("[INFO] 设备信息：") + str(get_device()))

    # fix model_path when version is lower than 1.5
    model_name = ["denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt", "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", "deverb_bs_roformer_8_256dim_8depth.ckpt", "deverb_bs_roformer_8_384dim_10depth.ckpt", "deverb_mel_band_roformer_ep_27_sdr_10.4567.ckpt"]
    for model in model_name:
        if os.path.exists(os.path.join(MODEL_FOLDER, "vocal_models", model)):
            shutil.move(os.path.join(MODEL_FOLDER, "vocal_models", model), os.path.join(MODEL_FOLDER, "single_stem_models", model))

def webui_restart():
    os.execl(PYTHON, PYTHON, *sys.argv)

def i18n(key):
    try:
        config = load_configs(WEBUI_CONFIG)
        if config["settings"]["language"]== "Auto":
            language = locale.getdefaultlocale()[0]
        else: language = config['settings']['language']
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
            if len(gpus) == 1:
                return gpus[0]
            return gpus
        elif backends.mps.is_available():
            return i18n("使用MPS")
        else:
            return i18n("无可用的加速设备, 使用CPU")
    except Exception:
        return i18n("设备检查失败")

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

def print_command(command):
    print("\033[32m" + "Use command: " + command + "\033[0m")

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
                downloaded_model.append(files)
        return downloaded_model
    return [i18n("请选择模型类型")]

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
        else: main_link = "huggingface.co"
    for keys in config.keys():
        for model in config[keys]:
            if model["name"] == model_name:
                model_type = model["model_type"]
                model_path = os.path.join(MODEL_FOLDER, keys, model_name)
                config_path = model["config_path"]
                download_link = model["link"]
                try:
                    download_link = download_link.replace("huggingface.co", main_link)
                except: pass
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
            downloaded_model.append(files)
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
        else: main_link = "huggingface.co"
    for keys in config.keys():
        if keys == model:
            primary_stem = config[keys]["primary_stem"]
            secondary_stem = config[keys]["secondary_stem"]
            model_url = config[keys]["download_link"]
            try:
                model_url = model_url.replace("huggingface.co", main_link)
            except: pass
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
    if not os.path.exists(folder):
        os.makedirs(folder)
    absolute_path = os.path.abspath(folder)
    if platform.system() == "Windows":
        os.system(f"explorer {absolute_path}")
    elif platform.system() == "Darwin":
        os.system(f"open {absolute_path}")
    else:
        os.system(f"xdg-open {absolute_path}")

def save_training_config(train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers, train_device_ids, train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_accelerate, train_pre_validate):
    config = load_configs(WEBUI_CONFIG)
    config['training']['model_type'] = train_model_type
    config['training']['config_path'] = train_config_path
    config['training']['dataset_type'] = train_dataset_type
    config['training']['dataset_path'] = train_dataset_path
    config['training']['valid_path'] = train_valid_path
    config['training']['num_workers'] = train_num_workers
    config['training']['device_ids'] = train_device_ids
    config['training']['seed'] = train_seed
    config['training']['pin_memory'] = train_pin_memory
    config['training']['use_multistft_loss'] = train_use_multistft_loss
    config['training']['use_mse_loss'] = train_use_mse_loss
    config['training']['use_l1_loss'] = train_use_l1_loss
    config['training']['accelerate'] = train_accelerate
    config['training']['pre_valid'] = train_pre_validate
    config['training']['results_path'] = train_results_path
    save_configs(config, WEBUI_CONFIG)
    return i18n("配置保存成功!")

def save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
    config = load_configs(WEBUI_CONFIG)
    config['inference']['vr_select_model'] = vr_select_model
    config['inference']['vr_window_size'] = vr_window_size
    config['inference']['vr_aggression'] = vr_aggression
    config['inference']['vr_output_format'] = vr_output_format
    config['inference']['vr_use_cpu'] = vr_use_cpu
    config['inference']['vr_primary_stem_only'] = vr_primary_stem_only
    config['inference']['vr_secondary_stem_only'] = vr_secondary_stem_only
    config['inference']['vr_multiple_audio_input'] = vr_multiple_audio_input
    config['inference']['vr_store_dir'] = vr_store_dir
    config['inference']['vr_batch_size'] = vr_batch_size
    config['inference']['vr_normalization'] = vr_normalization
    config['inference']['vr_post_process_threshold'] = vr_post_process_threshold
    config['inference']['vr_invert_spect'] = vr_invert_spect
    config['inference']['vr_enable_tta'] = vr_enable_tta
    config['inference']['vr_high_end_process'] = vr_high_end_process
    config['inference']['vr_enable_post_process'] = vr_enable_post_process
    config['inference']['vr_debug_mode'] = vr_debug_mode
    save_configs(config, WEBUI_CONFIG)

def save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
    config = load_configs(WEBUI_CONFIG)
    config['inference']['selected_model'] = selected_model
    config['inference']['gpu_id'] = gpu_id
    config['inference']['output_format'] = output_format
    config['inference']['force_cpu'] = force_cpu
    config['inference']['extract_instrumental'] = extract_instrumental
    config['inference']['use_tta'] = use_tta
    config['inference']['store_dir'] = store_dir
    config['inference']['multiple_audio_input'] = input_folder
    save_configs(config, WEBUI_CONFIG)

def save_uvr_modeldir(select_uvr_model_dir):
    if not os.path.exists(select_uvr_model_dir):
        return i18n("请选择正确的模型目录")
    config = load_configs(WEBUI_CONFIG)
    config['settings']['uvr_model_dir'] = select_uvr_model_dir
    save_configs(config, WEBUI_CONFIG)
    return i18n("设置保存成功! 请重启WebUI以应用。")

def reset_settings():
    try:
        config = load_configs(WEBUI_CONFIG)
        config_backup = load_configs(WEBUI_CONFIG_BACKUP)
        for key in config_backup['settings'][key]:
            config['settings'][key] = config_backup['settings'][key]
        save_configs(config, WEBUI_CONFIG)
        return i18n("设置重置成功, 请重启WebUI刷新! ")
    except Exception as e:
        print(e)
        return i18n("设置重置失败!")

def reset_webui_config():
    try:
        config = load_configs(WEBUI_CONFIG)
        config_backup = load_configs(WEBUI_CONFIG_BACKUP)
        for key in config_backup['training'][key]:
            config['training'][key] = config_backup['training'][key]
        for key in config_backup['inference'][key]:
            config['inference'][key] = config_backup['inference'][key]
        for key in config_backup['tools'][key]:
            config['tools'][key] = config_backup['tools'][key]
        save_configs(config, WEBUI_CONFIG)
        return i18n("记录重置成功, 请重启WebUI刷新! ")
    except Exception as e:
        print(e)
        return i18n("记录重置失败!")

def init_selected_model():
    try:
        batch_size, dim_t, num_overlap = i18n("该模型不支持修改此值"), i18n("该模型不支持修改此值"), i18n("该模型不支持修改此值")
        config = load_configs(WEBUI_CONFIG)
        selected_model = config['inference']['selected_model']
        _, config_path, _, _ = get_msst_model(selected_model)
        config = load_configs(config_path)
        if config.inference.get('batch_size'):
            batch_size = config.inference.get('batch_size')
        if config.inference.get('dim_t'):
            dim_t = config.inference.get('dim_t')
        if config.inference.get('num_overlap'):
            num_overlap = config.inference.get('num_overlap')
        return batch_size, dim_t, num_overlap
    except: return i18n("请先选择模型"), i18n("请先选择模型"), i18n("请先选择模型")

def init_selected_vr_model():
    webui_config = load_configs(WEBUI_CONFIG)
    config = load_configs(VR_MODEL)
    model = webui_config['inference']['vr_select_model']
    if not model:
        vr_primary_stem_only = i18n("仅输出主音轨")
        vr_secondary_stem_only = i18n("仅输出次音轨")
        return vr_primary_stem_only, vr_secondary_stem_only
    primary_stem, secondary_stem, _, _ = get_vr_model(model)
    vr_primary_stem_only = f"{primary_stem} Only"
    vr_secondary_stem_only = f"{secondary_stem} Only"
    return vr_primary_stem_only, vr_secondary_stem_only

def update_train_start_check_point(path):
    if not os.path.isdir(path):
        raise gr.Error(i18n("请先选择模型保存路径! "))
    ckpt_files = [f for f in os.listdir(path) if f.endswith(('.ckpt', '.chpt', '.th'))]
    return gr.Dropdown(label=i18n("初始模型"), choices=ckpt_files if ckpt_files else ["None"])

def update_selected_model(model_type):
    webui_config = load_configs(WEBUI_CONFIG)
    webui_config["inference"]["model_type"] = model_type
    save_configs(webui_config, WEBUI_CONFIG)
    return gr.Dropdown(label=i18n("选择模型"), choices=load_selected_model(), value=None, interactive=True, scale=4)

def update_inference_settings(selected_model):
    batch_size = gr.Textbox(label="batch_size", value=i18n("该模型不支持修改此值"), interactive=False)
    dim_t = gr.Textbox(label="dim_t", value=i18n("该模型不支持修改此值"), interactive=False)
    num_overlap = gr.Textbox(label="num_overlap", value=i18n("该模型不支持修改此值"), interactive=False)
    normalize = gr.Checkbox(label=i18n("normalize (该模型不支持修改此值) "), value=False, interactive=False)
    if selected_model and selected_model !="":
        _, config_path, _, _ = get_msst_model(selected_model)
        config = load_configs(config_path)
        if config.inference.get('batch_size'):
            batch_size = gr.Textbox(label="batch_size", value=str(config.inference.get('batch_size')), interactive=True)
        if config.inference.get('dim_t'):
            dim_t = gr.Textbox(label="dim_t", value=str(config.inference.get('dim_t')), interactive=True)
        if config.inference.get('num_overlap'):
            num_overlap = gr.Textbox(label="num_overlap", value=str(config.inference.get('num_overlap')), interactive=True)
        if config.inference.get('normalize'):
            normalize = gr.Checkbox(label="normalize", value=config.inference.get('normalize'), interactive=True)
    return batch_size, dim_t, num_overlap, normalize

def save_config(selected_model, batch_size, dim_t, num_overlap, normalize):
    _, config_path, _, _ = get_msst_model(selected_model)
    config = load_configs(config_path)
    if config.inference.get('batch_size'):
        config.inference['batch_size'] = int(batch_size) if batch_size.isdigit() else None
    if config.inference.get('dim_t'):
        config.inference['dim_t'] = int(dim_t) if dim_t.isdigit() else None
    if config.inference.get('num_overlap'):
        config.inference['num_overlap'] = int(num_overlap) if num_overlap.isdigit() else None
    if config.inference.get('normalize'):
        config.inference['normalize'] = normalize
    save_configs(config, config_path)
    return i18n("配置保存成功!")

def reset_config(selected_model):
    _, original_config_path, _, _ = get_msst_model(selected_model)
    if original_config_path.startswith(UNOFFICIAL_MODEL):
        return i18n("非官方模型不支持重置配置!")
    dir_path, file_name = os.path.split(original_config_path)
    backup_dir_path = dir_path.replace("configs", "configs_backup", 1)
    backup_config_path = os.path.join(backup_dir_path, file_name)
    if os.path.exists(backup_config_path):
        shutil.copy(backup_config_path, original_config_path)
        update_inference_settings(selected_model)
        return i18n("配置重置成功!")
    else:
        return i18n("备份配置文件不存在!")

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
            print(i18n("已停止进程"))

def run_inference_single(selected_model, input_audio, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
    input_folder = None
    if not input_audio:
        return i18n("请上传一个音频文件。")
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)
    os.makedirs(TEMP_PATH)
    shutil.copy(input_audio, TEMP_PATH)
    input_path = TEMP_PATH
    save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
    run_inference(selected_model, input_path, store_dir,extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
    return i18n("处理完成! 分离完成的音频文件已保存在") + store_dir

def run_multi_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
    save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
    run_inference(selected_model, input_folder, store_dir,extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
    return i18n("处理完成! 分离完成的音频文件已保存在") + store_dir

def run_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta, extra_store_dir=None):
    if not bool(re.match(r'^(\d+)(?:\s(?!\1)\d+)*$', gpu_id)):
        raise gr.Error(i18n("GPU ID格式错误, 请重新输入"))
    if selected_model == "":
        raise gr.Error(i18n("请选择模型"))
    if input_folder == "":
        raise gr.Error(i18n("请选择输入目录"))
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    if extra_store_dir and not os.path.exists(extra_store_dir):
        os.makedirs(extra_store_dir)
    start_check_point, config_path, model_type, _ = get_msst_model(selected_model)
    gpu_ids = gpu_id if not force_cpu else "0"
    extract_instrumental_option = "--extract_instrumental" if extract_instrumental else ""
    force_cpu_option = "--force_cpu" if force_cpu else ""
    use_tta_option = "--use_tta" if use_tta else ""
    extra_store_dir = f"--extra_store_dir \"{extra_store_dir}\"" if extra_store_dir else ""
    command = f"{PYTHON} msst_inference.py --model_type {model_type} --config_path \"{config_path}\" --start_check_point \"{start_check_point}\" --input_folder \"{input_folder}\" --store_dir \"{store_dir}\" --device_ids {gpu_ids} --output_format {output_format} {extract_instrumental_option} {force_cpu_option} {use_tta_option} {extra_store_dir}"
    msst_inference = threading.Thread(target=run_command, args=(command,), name="msst_inference")
    msst_inference.start()
    msst_inference.join()

def vr_inference_single(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_single_audio, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
    vr_multiple_audio_input = None
    if not os.path.isfile(vr_single_audio):
        return i18n("请上传一个音频文件")
    if not vr_select_model:
        return i18n("请选择模型")
    if not vr_store_dir:
        return i18n("请选择输出目录")
    if not os.path.exists(vr_store_dir):
        os.makedirs(vr_store_dir)
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)
    os.makedirs(TEMP_PATH)
    shutil.copy(vr_single_audio, TEMP_PATH)
    vr_single_audio = os.path.join(TEMP_PATH, os.path.basename(vr_single_audio))
    save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
    vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_single_audio, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
    return i18n("处理完成, 结果已保存至") + vr_store_dir

def vr_inference_multi(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
    if not os.path.isdir(vr_multiple_audio_input):
        return i18n("请选择输入文件夹")
    if not vr_select_model:
        return i18n("请选择模型")
    if not vr_store_dir:
        return i18n("请选择输出目录")
    if not os.path.exists(vr_store_dir):
        os.makedirs(vr_store_dir)
    save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
    vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
    return i18n("处理完成, 结果已保存至") + vr_store_dir

def vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode, save_another_stem=False, extra_output_dir=None):
    primary_stem, secondary_stem, _, model_file_dir = get_vr_model(vr_select_model)
    audio_file = vr_audio_input
    model_filename = vr_select_model
    output_format = vr_output_format
    output_dir = vr_store_dir
    invert_spect = "--invert_spect" if vr_invert_spect else ""
    normalization = vr_normalization
    debug_mode = "--debug" if vr_debug_mode else ""
    if vr_primary_stem_only:
        if vr_secondary_stem_only:
            single_stem = ""
        else:
            single_stem = f"--single_stem \"{primary_stem}\""
    else:
        if vr_secondary_stem_only:
            single_stem = f"--single_stem \"{secondary_stem}\""
        else:
            single_stem = ""
    vr_batch_size = int(vr_batch_size)
    vr_aggression = int(vr_aggression)
    use_cpu = "--use_cpu" if vr_use_cpu else ""
    vr_enable_tta = "--vr_enable_tta" if vr_enable_tta else ""
    vr_high_end_process = "--vr_high_end_process" if vr_high_end_process else ""
    vr_enable_post_process = "--vr_enable_post_process" if vr_enable_post_process else ""
    save_another_stem = "--save_another_stem" if save_another_stem else ""
    extra_output_dir = f"--extra_output_dir \"{extra_output_dir}\"" if extra_output_dir else ""
    command = f"{PYTHON} uvr_inference.py \"{audio_file}\" {debug_mode} --model_filename \"{model_filename}\" --output_format {output_format} --output_dir \"{output_dir}\" --model_file_dir \"{model_file_dir}\" {invert_spect} --normalization {normalization} {single_stem} {use_cpu} --vr_batch_size {vr_batch_size} --vr_window_size {vr_window_size} --vr_aggression {vr_aggression} {vr_enable_tta} {vr_high_end_process} {vr_enable_post_process} --vr_post_process_threshold {vr_post_process_threshold} {save_another_stem} {extra_output_dir}"
    vr_inference = threading.Thread(target=run_command, args=(command,), name="vr_inference")
    vr_inference.start()
    vr_inference.join()

def update_model_name(model_type):
    if model_type == "UVR_VR_Models":
        model_map = load_vr_model()
        return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)
    else:
        model_map = load_selected_model(model_type)
        return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)

def update_model_stem(model_type, model_name):
    if model_type == "UVR_VR_Models":
        primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
        return gr.Dropdown(label=i18n("输出音轨"), choices=[primary_stem, secondary_stem], interactive=True)
    else:
        return gr.Dropdown(label=i18n("输出音轨"), choices=["primary_stem"], value="primary_stem", interactive=False)

def add_to_flow_func(model_type, model_name, stem, secondary_output, df):
    if not model_type or not model_name:
        return df
    if model_type == "UVR_VR_Models" and not stem:
        return df
    if model_type == "UVR_VR_Models" and stem == "primary_stem":
        return df
    if not secondary_output or secondary_output == "":
        secondary_output = "False"
    new_data = pd.DataFrame({"model_type": [model_type], "model_name": [model_name], "stem": [stem], "secondary_output": [secondary_output]})
    if df["model_type"].iloc[0] == "" or df["model_name"].iloc[0] == "":
        return new_data
    updated_df = pd.concat([df, new_data], ignore_index=True)
    return updated_df

def save_flow_func(preset_name, df):
    preset_data = load_configs(PRESETS)
    if preset_name is None or preset_name == "":
        output_message = i18n("请填写预设名称")
        preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=list(preset_data.keys()))
        preset_name_select = gr.Dropdown(label=i18n("请选择预设"), choices=list(preset_data.keys()))
        return output_message, preset_name_delete, preset_name_select
    preset_dict = {f"Step_{index + 1}": row.dropna().to_dict() for index, row in df.iterrows()}
    preset_data[preset_name] = preset_dict
    save_configs(preset_data, PRESETS)
    output_message = i18n("预设") + preset_name + i18n("保存成功")
    preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=list(preset_data.keys()))
    preset_name_select = gr.Dropdown(label=i18n("请选择预设"), choices=list(preset_data.keys()))
    return output_message, preset_name_delete, preset_name_select

def reset_flow_func():
    return gr.Dataframe(pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""]}), interactive=False, label=None)

def load_preset(preset_name):
    preset_data = load_configs(PRESETS)
    if preset_name in preset_data.keys():
        preset_flow = pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""], "secondary_output": [""]})
        preset_dict = preset_data[preset_name]
        for key in preset_dict.keys():
            try:
                secondary_output = preset_dict[key]["secondary_output"]
            except KeyError:
                secondary_output = "False"
            preset_flow = add_to_flow_func(preset_dict[key]["model_type"], preset_dict[key]["model_name"], preset_dict[key]["stem"], secondary_output, preset_flow)
        return preset_flow
    return gr.Dataframe(pd.DataFrame({"model_type": [i18n("预设不存在")], "model_name": [i18n("预设不存在")], "stem": [i18n("预设不存在")], "secondary_output": [i18n("预设不存在")]}), interactive=False, label=None)

def delete_func(preset_name):
    preset_data = load_configs(PRESETS)
    if preset_name in preset_data.keys():
        _, select_preset_backup = backup_preset_func()
        del preset_data[preset_name]
        save_configs(preset_data, PRESETS)
        output_message = i18n("预设") + preset_name + i18n("删除成功")
        preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=list(preset_data.keys()))
        preset_name_select = gr.Dropdown(label=i18n("请选择预设"), choices=list(preset_data.keys()))
        preset_flow_delete = gr.Dataframe(pd.DataFrame({"model_type": [i18n("预设已删除")], "model_name": [i18n("预设已删除")], "stem": [i18n("预设已删除")], "secondary_output": [i18n("预设已删除")]}), interactive=False, label=None)
        return output_message, preset_name_delete, preset_name_select, preset_flow_delete, select_preset_backup
    else:
        return i18n("预设不存在")

def run_single_inference_flow(input_audio, store_dir, preset_name, force_cpu, output_format_flow):
    if not input_audio:
        return i18n("请上传一个音频文件")
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)
    os.makedirs(TEMP_PATH)
    shutil.copy(input_audio, TEMP_PATH)
    input_folder = TEMP_PATH
    msg = run_inference_flow(input_folder, store_dir, preset_name, force_cpu, output_format_flow, isSingle=True)
    return msg

def run_inference_flow(input_folder, store_dir, preset_name, force_cpu, output_format_flow, isSingle=False):
    global stop_infer_flow
    stop_infer_flow = False
    start_time = time.time()
    preset_data = load_configs(PRESETS)
    if not preset_name in preset_data.keys():
        return i18n("预设") + preset_name + i18n("不存在")
    config = load_configs(WEBUI_CONFIG)
    config['inference']['preset'] = preset_name
    config['inference']['force_cpu'] = force_cpu
    config['inference']['output_format_flow'] = output_format_flow
    config['inference']['input_folder_flow'] = input_folder
    config['inference']['store_dir_flow'] = store_dir
    save_configs(config, WEBUI_CONFIG)
    model_list = preset_data[preset_name]
    input_to_use = input_folder
    if os.path.exists(TEMP_PATH) and not isSingle:
        shutil.rmtree(TEMP_PATH)
    tmp_store_dir = f"{TEMP_PATH}/inferflow_step1_output"
    for step in model_list.keys():
        model_name = model_list[step]["model_name"]
        if model_name not in load_msst_model() and model_name not in load_vr_model():
            return i18n("模型") + model_name + i18n("不存在")
    i = 0
    for step in model_list.keys():
        if stop_infer_flow:
            stop_infer_flow = False
            break
        if i == 0:
            input_to_use = input_folder
        elif i < len(model_list.keys()) - 1 and i > 0:
            if input_to_use != input_folder:
                shutil.rmtree(input_to_use)
            input_to_use = tmp_store_dir
            tmp_store_dir = f"{TEMP_PATH}/inferflow_step{i+1}_output"
        elif i == len(model_list.keys()) - 1:
            input_to_use = tmp_store_dir
            tmp_store_dir = store_dir
        console = Console()
        model_name = model_list[step]["model_name"]
        console.rule(f"[yellow]Step {i+1}: Running inference using {model_name}", style="yellow")
        if model_list[step]["model_type"] == "UVR_VR_Models":
            primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
            stem = model_list[step]["stem"]
            vr_select_model = model_name
            vr_window_size = config['inference']['vr_window_size']
            vr_aggression = config['inference']['vr_aggression']
            vr_output_format = output_format_flow
            vr_use_cpu = force_cpu
            vr_primary_stem_only = True if stem == primary_stem else False
            vr_secondary_stem_only = True if stem == secondary_stem else False
            vr_audio_input = input_to_use
            vr_store_dir = tmp_store_dir
            vr_batch_size = config['inference']['vr_batch_size']
            vr_normalization = config['inference']['vr_normalization']
            vr_post_process_threshold = config['inference']['vr_post_process_threshold']
            vr_invert_spect = config['inference']['vr_invert_spect']
            vr_enable_tta = config['inference']['vr_enable_tta']
            vr_high_end_process = config['inference']['vr_high_end_process']
            vr_enable_post_process = config['inference']['vr_enable_post_process']
            vr_debug_mode = config['inference']['vr_debug_mode']
            try:
                secondary_output = model_list[step]["secondary_output"]
            except KeyError:
                secondary_output = "False"
            if secondary_output == "True":
                save_another_stem = True
                extra_output_dir = os.path.join(store_dir, "secondary_output")
            else:
                save_another_stem = False
                extra_output_dir = None
            vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode, save_another_stem, extra_output_dir)
        else:
            gpu_id = config['inference']['gpu_id'] if not force_cpu else "0"
            use_tta = config['inference']['use_tta']
            try:
                secondary_output = model_list[step]["secondary_output"]
            except KeyError:
                secondary_output = "False"
            if secondary_output == "True":
                extract_instrumental = True
                extra_store_dir = os.path.join(store_dir, "secondary_output")
            else:
                extract_instrumental = False
                extra_store_dir = None
            run_inference(model_name, input_to_use, tmp_store_dir, extract_instrumental, gpu_id, output_format_flow, force_cpu, use_tta, extra_store_dir)
        i += 1
    shutil.rmtree(TEMP_PATH)
    finish_time = time.time()
    elapsed_time = finish_time - start_time
    Console().rule(f"[yellow]Finished runing {preset_name}! Costs {elapsed_time:.2f}s", style="yellow")
    return i18n("处理完成! 分离完成的音频文件已保存在") + store_dir

def preset_backup_list():
    backup_dir = BACKUP
    if not os.path.exists(backup_dir):
        return [i18n("暂无备份文件")]
    backup_files = []
    for file in os.listdir(backup_dir):
        if file.startswith("preset_backup_") and file.endswith(".json"):
            backup_files.append(file)
    if backup_files == []:
        return [i18n("暂无备份文件")]
    return backup_files

def restore_preset_func(backup_file):
    backup_dir = BACKUP
    if not backup_file or backup_file == i18n("暂无备份文件"):
        return i18n("请选择备份文件")
    backup_data = load_configs(os.path.join(backup_dir, backup_file))
    save_configs(backup_data, PRESETS)
    output_message_manage = i18n("已成功恢复备份") + backup_file
    preset_dropdown = gr.Dropdown(label=i18n("请选择预设"), choices=list(backup_data.keys()))
    preset_name_delet = gr.Dropdown(label=i18n("请选择预设"), choices=list(backup_data.keys()))
    preset_flow_delete = pd.DataFrame({"model_type": [i18n("请先选择预设")], "model_name": [i18n("请先选择预设")], "stem": [i18n("请先选择预设")], "secondary_output": [i18n("请先选择预设")]})
    return output_message_manage, preset_dropdown, preset_name_delet, preset_flow_delete

def backup_preset_func():
    backup_dir = BACKUP
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    backup_file = f"preset_backup_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    preset_data = load_configs(PRESETS)
    save_configs(preset_data, os.path.join(backup_dir, backup_file))
    msg = i18n("已成功备份至") + backup_file
    select_preset_backup = gr.Dropdown(label=i18n("选择需要恢复的预设流程备份"), choices=preset_backup_list(), interactive=True, scale=4)
    return msg, select_preset_backup

def convert_audio(uploaded_files, ffmpeg_output_format, ffmpeg_output_folder):
    if not uploaded_files:
        return i18n("请上传至少一个文件")
    success_files = []
    for uploaded_file in uploaded_files:
        uploaded_file_path = uploaded_file.name
        output_path = ffmpeg_output_folder
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        config = load_configs(WEBUI_CONFIG)
        config['tools']['ffmpeg_output_format'] = ffmpeg_output_format
        config['tools']['ffmpeg_output_folder'] = ffmpeg_output_folder
        save_configs(config, WEBUI_CONFIG)
        output_file = os.path.join(output_path, os.path.splitext(
            os.path.basename(uploaded_file_path))[0] + "." + ffmpeg_output_format)
        command = f"{FFMPEG} -i \"{uploaded_file_path}\" \"{output_file}\""
        print_command(command)
        try:
            subprocess.run(command, shell=True, check=True)
            success_files.append(output_file)
        except subprocess.CalledProcessError:
            print(f"Fail to convert file: {uploaded_file_path}\n")
            continue
    if not success_files:
        return i18n("所有文件转换失败, 请检查文件格式和ffmpeg路径。")
    else:
        text = i18n("处理完成, 文件已保存为: ") + "\n" + "\n".join(success_files)
        return text

def merge_audios(input_folder, output_folder):
    config = load_configs(WEBUI_CONFIG)
    config['tools']['merge_audio_input'] = input_folder
    config['tools']['merge_audio_output'] = output_folder
    save_configs(config, WEBUI_CONFIG)
    combined_audio = AudioSegment.empty()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, "merged_audio.wav")
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):
            file_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_file(file_path)
            combined_audio += audio
    try:
        combined_audio.export(output_file, format="wav")
        return i18n("处理完成, 文件已保存为: ") + output_file
    except Exception as e:
        print(e)
        return i18n("处理失败!")

def read_and_resample_audio(file_path, target_sr=44100):
    audio, _ = librosa.load(file_path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = np.vstack((audio, audio))
    elif audio.ndim == 2 and audio.shape[0] == 1:
        audio = np.vstack((audio[0], audio[0]))
    return audio

def match_length(ref_audio, est_audio):
    min_length = min(ref_audio.shape[1], est_audio.shape[1])
    ref_audio = ref_audio[:, :min_length]
    est_audio = est_audio[:, :min_length]
    return ref_audio, est_audio

def compute_sdr(reference, estimated):
    sdr, _, _, _ = bss_eval_sources(reference, estimated)
    return sdr

def process_audio(true_path, estimated_path):
    target_sr = 44100
    true_audio = read_and_resample_audio(
        true_path, target_sr)
    estimated_audio = read_and_resample_audio(
        estimated_path, target_sr)
    true_audio, estimated_audio = match_length(true_audio, estimated_audio)
    sdr = compute_sdr(true_audio, estimated_audio)
    print(f"SDR: {sdr}")
    return sdr

def ensemble(files, ensemble_mode, weights, output_path):
    if len(files) < 2:
        return i18n("请上传至少2个文件")
    if len(files) != len(weights.split()):
        return i18n("上传的文件数目与权重数目不匹配")
    else:
        config = load_configs(WEBUI_CONFIG)
        config['tools']['ensemble_type'] = ensemble_mode
        config['tools']['ensemble_output_folder'] = output_path
        save_configs(config, WEBUI_CONFIG)
        files_argument = " ".join(files)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, f"ensemble_{ensemble_mode}.wav")
        command = f"{PYTHON} ensemble.py --files {files_argument} --type {ensemble_mode} --weights {weights} --output {output_path}"
        print_command(command)
        try:
            subprocess.run(command, shell = True)
            return i18n("处理完成, 文件已保存为: ") + output_path
        except Exception as e:
            return i18n("处理失败!")

def some_inference(audio_file, bpm, output_dir):
    model = "tools/SOME_weights/model_steps_64000_simplified.ckpt"
    if not os.path.isfile(model):
        return i18n("请先下载SOME预处理模型并放置在tools/SOME_weights文件夹下! ")
    if not audio_file.endswith('.wav'):
        return i18n("请上传wav格式文件")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config = load_configs(WEBUI_CONFIG)
    config['tools']['some_output_folder'] = output_dir
    save_configs(config, WEBUI_CONFIG)
    tempo = int(bpm)
    file_name = os.path.basename(audio_file)[0:-4]
    midi = os.path.join(output_dir, f"{file_name}.mid")
    command = f"{PYTHON} tools/SOME/infer.py --model {model} --wav \"{audio_file}\" --midi \"{midi}\" --tempo {tempo}"
    print_command(command)
    try:
        subprocess.run(command, shell=True)
        return i18n("处理完成, 文件已保存为: ") + midi
    except Exception as e:
        return i18n("处理失败!")

def upgrade_download_model_name(model_type_dropdown):
    if model_type_dropdown == "UVR_VR_Models":
        model_map = load_configs(VR_MODEL)
        return gr.Dropdown(label=i18n("选择模型"), choices=[keys for keys in model_map.keys()])
    else:
        model_map = load_configs(MSST_MODEL)
        return gr.Dropdown(label=i18n("选择模型"), choices=[model["name"] for model in model_map[model_type_dropdown]])

def download_model(model_type, model_name):
    if model_type not in MODEL_CHOICES:
        return i18n("请选择模型类型")
    if model_type == "UVR_VR_Models":
        downloaded_model = load_vr_model()
        if model_name in downloaded_model:
            return i18n("模型") + model_name + i18n("已安装")
        _, _, model_url, model_path = get_vr_model(model_name)
        os.makedirs(model_path, exist_ok=True)
        return download_file(model_url, os.path.join(model_path, model_name), model_name)
    else:
        presets = load_configs(MSST_MODEL)
        model_mapping = load_msst_model()
        if model_name in model_mapping:
            return i18n("模型") + model_name + i18n("已安装")
        if model_type not in presets:
            return i18n("模型类型") + model_type + i18n("不存在")
        _, _, _, model_url = get_msst_model(model_name)
        model_path = f"pretrain/{model_type}/{model_name}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        return download_file(model_url, model_path, model_name)

def download_file(url, path, model_name):
    try:
        print(i18n("模型") + model_name + i18n("下载中"))
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True
            ) as progress_bar:
                last_update_time = time.time()
                bytes_written = 0
                for data in r.iter_content(1024):
                    f.write(data)
                    bytes_written += len(data)
                    current_time = time.time()
                    if current_time - last_update_time >= 1:
                        progress_bar.update(bytes_written)
                        bytes_written = 0
                        last_update_time = current_time
                progress_bar.update(bytes_written)
        return i18n("模型") + model_name + i18n("下载成功")
    except Exception as e:
        print(e)
        return i18n("模型") + model_name + i18n("下载失败")

def manual_download_model(model_type, model_name):
    if model_type not in MODEL_CHOICES:
        return i18n("请选择模型类型")
    if model_type == "UVR_VR_Models":
        downloaded_model = load_vr_model()
        if model_name in downloaded_model:
            return i18n("模型") + model_name + i18n("已安装")
        _, _, model_url, _ = get_vr_model(model_name)
    else:
        presets = load_configs(MSST_MODEL)
        model_mapping = load_msst_model()
        if model_name in model_mapping:
            return i18n("模型") + model_name + i18n("已安装")
        if model_type not in presets:
            return i18n("模型类型") + model_type + i18n("不存在")
        _, _, _, model_url = get_msst_model(model_name)
    webbrowser.open(model_url)
    return i18n("已打开") + model_name + i18n("的下载链接")

def update_vr_param(is_BV_model, is_VR51_model, model_param):
    balance_value = gr.Number(label="balance_value", value=0.0, minimum=0.0, maximum=0.9, step=0.1, interactive=True, visible=True if is_BV_model else False)
    out_channels = gr.Number(label="Out Channels", value=32, minimum=1, step=1, interactive=True, visible=True if is_VR51_model else False)
    out_channels_lstm = gr.Number(label="Out Channels (LSTM layer)", value=128, minimum=1, step=1, interactive=True, visible=True if is_VR51_model else False)
    upload_param = gr.File(label=i18n("上传参数文件"), type="filepath", interactive=True, visible=True if model_param == i18n("上传参数") else False)
    return balance_value, out_channels, out_channels_lstm, upload_param

def install_unmsst_model(unmsst_model, unmsst_config, unmodel_class, unmodel_type, unmsst_model_link):
    if not os.path.exists(os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json")):
        os.makedirs(UNOFFICIAL_MODEL, exist_ok=True)
    try:
        model_map = load_configs(os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json"))
    except FileNotFoundError:
        model_map = {"multi_stem_models": [], "single_stem_models": [], "vocal_models": []}
    try:
        if unmsst_model.endswith((".ckpt", ".chpt", ".th")):
            shutil.copy(unmsst_model, os.path.join(MODEL_FOLDER, unmodel_class))
        else: return i18n("请上传'ckpt', 'chpt', 'th'格式的模型文件")
        if unmsst_config.endswith(".yaml"):
            shutil.copy(unmsst_config, UNOFFICIAL_MODEL)
        else: return i18n("请上传'.yaml'格式的配置文件")
        config = {
            "name": os.path.basename(unmsst_model),
            "config_path": os.path.join(UNOFFICIAL_MODEL, os.path.basename(unmsst_config)),
            "model_type": unmodel_type,
            "link": unmsst_model_link
        }
        model_map[unmodel_class].append(config)
        save_configs(model_map, os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json"))
        return i18n("模型") + os.path.basename(unmsst_model) + i18n("安装成功。重启WebUI以刷新模型列表")
    except Exception as e:
        print(e)
        return i18n("模型") + os.path.basename(unmsst_model) + i18n("安装失败")

def install_unvr_model(unvr_model, unvr_primary_stem, unvr_secondary_stem, model_param, is_karaoke_model, is_BV_model, is_VR51_model, balance_value, out_channels, out_channels_lstm, upload_param, unvr_model_link):
    if not os.path.exists(os.path.join(UNOFFICIAL_MODEL, "unofficial_vr_model.json")):
        os.makedirs(UNOFFICIAL_MODEL, exist_ok=True)
    try:
        model_map = load_configs(os.path.join(UNOFFICIAL_MODEL, "unofficial_vr_model.json"))
    except FileNotFoundError:
        model_map = {}
    try:
        if unvr_model.endswith(".pth"):
            shutil.copy(unvr_model, "pretrain/VR_Models")
        else: return i18n("请上传'.pth'格式的模型文件")
        if unvr_primary_stem != "" and unvr_secondary_stem != "" and unvr_primary_stem != unvr_secondary_stem:
            model_name = os.path.basename(unvr_model)
            model_map[model_name] = {}
            model_map[model_name]["model_path"] = os.path.join(MODEL_FOLDER, "VR_Models", model_name)
            model_map[model_name]["primary_stem"] = unvr_primary_stem
            model_map[model_name]["secondary_stem"] = unvr_secondary_stem
            model_map[model_name]["download_link"] = unvr_model_link
        else: return i18n("请输入正确的音轨名称")
        if model_param == i18n("上传参数"):
            if upload_param.endswith(".json"):
                shutil.copy(upload_param, "models/vocal_remover/uvr_lib_v5/vr_network/modelparams")
                model_map[model_name]["vr_model_param"] = os.path.basename(upload_param)[:-5]
            else: return i18n("请上传'.json'格式的参数文件")
        else: model_map[model_name]["vr_model_param"] = model_param
        if is_karaoke_model:
            model_map[model_name]["is_karaoke"] = True
        if is_BV_model:
            model_map[model_name]["is_bv_model"] = True
            model_map[model_name]["is_bv_model_rebalanced"] = balance_value
        if is_VR51_model:
            model_map[model_name]["nout"] = out_channels
            model_map[model_name]["nout_lstm"] = out_channels_lstm
        save_configs(model_map, os.path.join(UNOFFICIAL_MODEL, "unofficial_vr_model.json"))
        return i18n("模型") + os.path.basename(unvr_model) + i18n("安装成功。重启WebUI以刷新模型列表")
    except Exception as e:
        print(e)
        return i18n("模型") + os.path.basename(unvr_model) + i18n("安装失败")

def get_all_model_param():
    model_param = [i18n("上传参数")]
    for file in os.listdir("models/vocal_remover/uvr_lib_v5/vr_network/modelparams"):
        if file.endswith(".json"):
            model_param.append(file[:-5])
    return model_param

def start_training(train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers, train_device_ids, train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_start_check_point, train_accelerate, train_pre_validate):
    model_type = train_model_type
    config_path = train_config_path
    start_check_point = train_start_check_point
    results_path = train_results_path
    data_path = train_dataset_path
    dataset_type = train_dataset_type
    valid_path = train_valid_path
    num_workers = int(train_num_workers)
    device_ids = train_device_ids
    seed = int(train_seed)
    pin_memory = train_pin_memory
    use_multistft_loss = "--use_multistft_loss" if train_use_multistft_loss else ""
    use_mse_loss = "--use_mse_loss" if train_use_mse_loss else ""
    use_l1_loss = "--use_l1_loss" if train_use_l1_loss else ""
    pre_valid = "--pre_valid" if (train_accelerate and train_pre_validate) else ""
    if train_accelerate:
        train_file = "train_accelerate.py"
    else:
        train_file = "train.py"
    if model_type not in MODEL_TYPE:
        return i18n("模型类型错误, 请重新选择")
    if not os.path.isfile(config_path):
        return i18n("配置文件不存在, 请重新选择")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(data_path):
        return i18n("数据集路径不存在, 请重新选择")
    if not os.path.exists(valid_path):
        return i18n("验证集路径不存在, 请重新选择")
    if dataset_type not in [1, 2, 3, 4]:
        return i18n("数据集类型错误, 请重新选择")
    if not bool(re.match(r'^(\d+)(?:\s(?!\1)\d+)*$', device_ids)):
        return i18n("device_ids格式错误, 请重新输入")
    if train_start_check_point == "None" or train_start_check_point == "":
        start_check_point = ""
    elif os.path.exists(results_path):
        start_check_point = "--start_check_point " + "\"" + os.path.join(results_path, train_start_check_point) + "\""
    else:
        return i18n("模型保存路径不存在, 请重新选择")
    command = f"{PYTHON} {train_file} --model_type {model_type} --config_path \"{config_path}\" {start_check_point} --results_path \"{results_path}\" --data_path \"{data_path}\" --dataset_type {dataset_type} --valid_path \"{valid_path}\" --num_workers {num_workers} --device_ids {device_ids} --seed {seed} --pin_memory {pin_memory} {use_multistft_loss} {use_mse_loss} {use_l1_loss} {pre_valid}"
    threading.Thread(target=run_command, args=(command,), name="msst_training").start()
    return i18n("训练启动成功! 请前往控制台查看训练信息! ")

def validate_model(valid_model_type, valid_config_path, valid_model_path, valid_path, valid_results_path, valid_device_ids, valid_num_workers, valid_extension, valid_pin_memory):
    if valid_model_type not in MODEL_TYPE:
        return i18n("模型类型错误, 请重新选择")
    if not os.path.isfile(valid_config_path):
        return i18n("配置文件不存在, 请重新选择")
    if not os.path.isfile(valid_model_path):
        return i18n("模型不存在, 请重新选择")
    if not os.path.exists(valid_path):
        return i18n("验证集路径不存在, 请重新选择")
    if not os.path.exists(valid_results_path):
        os.makedirs(valid_results_path)
    if not bool(re.match(r'^(\d+)(?:\s(?!\1)\d+)*$', valid_device_ids)):
        return i18n("device_ids格式错误, 请重新输入")
    pin_memory = "--pin_memory" if valid_pin_memory else ""
    command = f"{PYTHON} valid.py --model_type {valid_model_type} --config_path \"{valid_config_path}\" --start_check_point \"{valid_model_path}\" --valid_path \"{valid_path}\" --store_dir \"{valid_results_path}\" --device_ids {valid_device_ids} --num_workers {valid_num_workers} --extension {valid_extension} {pin_memory}"
    msst_valid = threading.Thread(target=run_command, args=(command,), name="msst_valid")
    msst_valid.start()
    msst_valid.join()
    return i18n("验证完成! 请打开输出文件夹查看详细结果")

def check_webui_update():
    url = "https://github.com/SUC-DriverOld/MSST-WebUI/releases/latest"
    try:
        response = requests.get(url)
        response.raise_for_status()
        latest_version = response.url.split("/")[-1]
        if latest_version != PACKAGE_VERSION:
            return i18n("当前版本: ") + PACKAGE_VERSION + i18n(", 发现新版本: ") + latest_version
        else:
            return i18n("当前版本: ") + PACKAGE_VERSION + i18n(", 已是最新版本")
    except Exception:
        return i18n("检查更新失败")

def webui_goto_github():
    webbrowser.open("https://github.com/SUC-DriverOld/MSST-WebUI/releases/latest")

def change_language(language):
    config = load_configs(WEBUI_CONFIG)
    if language in language_dict.keys():
        config['settings']['language'] = language_dict[language]
    else:
        config['settings']['language'] = "Auto"
    save_configs(config, WEBUI_CONFIG)
    return i18n("语言已更改, 重启WebUI生效")

def get_language():
    config = load_configs(WEBUI_CONFIG)
    language = config['settings']['language']
    for key, value in language_dict.items():
        if value == language:
            return key
    return "Auto"

def save_port_to_config(port):
    port = int(port)
    config = load_configs(WEBUI_CONFIG)
    config['settings']['port'] = port
    save_configs(config, WEBUI_CONFIG)
    return i18n("成功将端口设置为") + str(port) + i18n(", 重启WebUI生效")

def change_download_link(link):
    config = load_configs(WEBUI_CONFIG)
    if link == i18n("huggingface.co (需要魔法)"):
        config['settings']['download_link'] = "huggingface.co"
    elif link == i18n("hf-mirror.com (镜像站可直连)"):
        config['settings']['download_link'] = "hf-mirror.com"
    else:
        config['settings']['download_link'] = "Auto"
    save_configs(config, WEBUI_CONFIG)
    return i18n("下载链接已更改")

def change_share_link(flag):
    config = load_configs(WEBUI_CONFIG)
    if flag:
        config['settings']['share_link'] = True
        save_configs(config, WEBUI_CONFIG)
        return i18n("公共链接已开启, 重启WebUI生效")
    else:
        config['settings']['share_link'] = False
        save_configs(config, WEBUI_CONFIG)
        return i18n("公共链接已关闭, 重启WebUI生效")

def change_local_link(flag):
    config = load_configs(WEBUI_CONFIG)
    if flag:
        config['settings']['local_link'] = True
        save_configs(config, WEBUI_CONFIG)
        return i18n("已开启局域网分享, 重启WebUI生效")
    else:
        config['settings']['local_link'] = False
        save_configs(config, WEBUI_CONFIG)
        return i18n("已关闭局域网分享, 重启WebUI生效")

setup_webui()
with gr.Blocks(
        theme=gr.Theme.load('tools/themes/theme_schema@1.2.2.json')
) as app:
    gr.Markdown(value=f"""### Music-Source-Separation-Training-Inference-Webui v{PACKAGE_VERSION}""")
    gr.Markdown(value=i18n("仅供个人娱乐和非商业用途, 禁止用于血腥/暴力/性相关/政治相关内容。[点击前往教程文档](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)<br>本整合包完全免费, 严禁以任何形式倒卖, 如果你从任何地方**付费**购买了本整合包, 请**立即退款**。<br> 整合包作者: [bilibili@阿狸不吃隼舞](https://space.bilibili.com/403335715) [Github@KitsuneX07](https://github.com/KitsuneX07) | [Bilibili@Sucial](https://space.bilibili.com/445022409) [Github@SUC-DriverOld](https://github.com/SUC-DriverOld) | Gradio主题: [Gradio Theme](https://huggingface.co/spaces/NoCrypt/miku)"))
    with gr.Tabs():
        webui_config = load_configs(WEBUI_CONFIG)
        presets = load_configs(PRESETS)
        models = load_configs(MSST_MODEL)
        vr_model = load_configs(VR_MODEL)

        with gr.TabItem(label=i18n("MSST分离")):
            gr.Markdown(value=i18n("MSST音频分离原项目地址: [https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)"))
            with gr.Row():
                select_model_type = gr.Dropdown(label=i18n("选择模型类型"), choices=["vocal_models", "multi_stem_models", "single_stem_models"], value=webui_config['inference']['model_type'] if webui_config['inference']['model_type'] else None, interactive=True, scale=1)
                selected_model = gr.Dropdown(label=i18n("选择模型"),choices=load_selected_model(),value=webui_config['inference']['selected_model'] if webui_config['inference']['selected_model'] else None,interactive=True,scale=4)
            with gr.Row():
                gpu_id = gr.Textbox(label=i18n("选择使用的GPU ID, 多卡用户请使用空格分隔GPU ID。可前往设置页面查看显卡信息。"),value=webui_config['inference']['gpu_id'] if webui_config['inference']['gpu_id'] else "0",interactive=True)
                output_format = gr.Dropdown(label=i18n("输出格式"),choices=["wav", "mp3", "flac"],value=webui_config['inference']['output_format'] if webui_config['inference']['output_format'] else "wav", interactive=True)
            with gr.Row():
                force_cpu = gr.Checkbox(label=i18n("使用CPU (注意: 使用CPU会导致速度非常慢) "),value=webui_config['inference']['force_cpu'] if webui_config['inference']['force_cpu'] else False,interactive=True)
                extract_instrumental = gr.Checkbox(label=i18n("同时输出次级音轨"),value=webui_config['inference']['extract_instrumental'] if webui_config['inference']['extract_instrumental'] else False,interactive=True)
                use_tta = gr.Checkbox(label=i18n("使用TTA (测试时增强), 可能会提高质量, 但速度稍慢"),value=webui_config['inference']['use_tta'] if webui_config['inference']['use_tta'] else False,interactive=True)
            with gr.Tabs():
                with gr.TabItem(label=i18n("单个音频上传")):
                    single_audio = gr.File(label=i18n("单个音频上传"), type="filepath")
                with gr.TabItem(label=i18n("批量音频上传")):
                    with gr.Row():
                        multiple_audio_input = gr.Textbox(label=i18n("输入目录"),value=webui_config['inference']['multiple_audio_input'] if webui_config['inference']['multiple_audio_input'] else "input/",interactive=True,scale=3)
                        select_multi_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
                        open_multi_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
            with gr.Row():
                store_dir = gr.Textbox(label=i18n("输出目录"),value=webui_config['inference']['store_dir'] if webui_config['inference']['store_dir'] else "results/",interactive=True,scale=3)
                select_store_btn = gr.Button(i18n("选择文件夹"), scale=1)
                open_store_btn = gr.Button(i18n("打开文件夹"), scale=1)
            with gr.Accordion(i18n("推理参数设置, 不同模型之间参数相互独立 (一般不需要动) "), open=False):
                gr.Markdown(value=i18n("只有在点击保存后才会生效。参数直接写入配置文件, 无法撤销。假如不知道如何设置, 请保持默认值。<br>请牢记自己修改前的参数数值, 防止出现问题以后无法恢复。请确保输入正确的参数, 否则可能会导致模型无法正常运行。<br>假如修改后无法恢复, 请点击``重置``按钮, 这会使得配置文件恢复到默认值。"))
                if webui_config['inference']['selected_model']:
                    batch_size_number, dim_t_number, num_overlap_number = init_selected_model()
                else:
                    batch_size_number, dim_t_number, num_overlap_number = i18n("请先选择模型"), i18n("请先选择模型"), i18n("请先选择模型")
                with gr.Row():
                    batch_size = gr.Textbox(label=i18n("batch_size: 批次大小, 一般不需要改"), value=batch_size_number)
                    dim_t = gr.Textbox(label=i18n("dim_t: 时序维度大小, 一般不需要改 (部分模型没有此参数)"), value=dim_t_number)
                    num_overlap = gr.Textbox(label=i18n("num_overlap: 窗口重叠长度, 数值越小速度越快, 但会牺牲效果"), value=num_overlap_number)
                normalize = gr.Checkbox(label=i18n("normalize: 是否对音频进行归一化处理 (部分模型没有此参数)"), value=False, interactive=False)
                reset_config_button = gr.Button(i18n("重置配置"), variant="secondary")
                save_config_button = gr.Button(i18n("保存配置"), variant="primary")
            with gr.Row():
                inference_single = gr.Button(i18n("单个音频分离"), variant="primary")
                inference_multiple = gr.Button(i18n("批量音频分离"), variant="primary")
            with gr.Row():
                output_message = gr.Textbox(label="Output Message", scale=4)
                stop_thread = gr.Button(i18n("强制停止"), scale=1)

            inference_single.click(fn=run_inference_single,inputs=[selected_model, single_audio, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta],outputs=output_message)
            inference_multiple.click(fn=run_multi_inference, inputs=[selected_model, multiple_audio_input, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta],outputs=output_message)
            select_model_type.change(fn=update_selected_model, inputs=[select_model_type], outputs=[selected_model])
            selected_model.change(fn=update_inference_settings,inputs=[selected_model],outputs=[batch_size, dim_t, num_overlap, normalize])
            save_config_button.click(fn=save_config,inputs=[selected_model, batch_size, dim_t, num_overlap, normalize],outputs=output_message)
            reset_config_button.click(fn=reset_config,inputs=[selected_model],outputs=output_message)
            select_store_btn.click(fn=select_folder, outputs=store_dir)
            open_store_btn.click(fn=open_folder, inputs=store_dir)
            select_multi_input_dir.click(fn=select_folder, outputs=multiple_audio_input)
            open_multi_input_dir.click(fn=open_folder, inputs=multiple_audio_input)
            stop_thread.click(fn=stop_all_thread)

        with gr.TabItem(label=i18n("UVR分离")):
            gr.Markdown(value=i18n("说明: 本整合包仅融合了UVR的VR Architecture模型, MDX23C和HtDemucs类模型可以直接使用前面的MSST音频分离。<br>UVR分离使用项目: [https://github.com/nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 并进行了优化。"))
            vr_select_model = gr.Dropdown(label=i18n("选择模型"), choices=load_vr_model(), value=webui_config['inference']['vr_select_model'] if webui_config['inference']['vr_select_model'] else None,interactive=True)
            with gr.Row():
                vr_window_size = gr.Dropdown(label=i18n("Window Size: 窗口大小, 用于平衡速度和质量"), choices=["320", "512", "1024"], value=webui_config['inference']['vr_window_size'] if webui_config['inference']['vr_window_size'] else "512", interactive=True)
                vr_aggression = gr.Number(label=i18n("Aggression: 主干提取强度, 范围-100-100, 人声请选5"), minimum=-100, maximum=100, value=webui_config['inference']['vr_aggression'] if webui_config['inference']['vr_aggression'] else 5, interactive=True)
                vr_output_format = gr.Dropdown(label=i18n("输出格式"), choices=["wav", "flac", "mp3"], value=webui_config['inference']['vr_output_format'] if webui_config['inference']['vr_output_format'] else "wav", interactive=True)
            with gr.Row():
                vr_primary_stem_label, vr_secondary_stem_label = init_selected_vr_model()
                vr_use_cpu = gr.Checkbox(label=i18n("使用CPU"), value=webui_config['inference']['vr_use_cpu'] if webui_config['inference']['vr_use_cpu'] else False, interactive=True)
                vr_primary_stem_only = gr.Checkbox(label=vr_primary_stem_label, value=webui_config['inference']['vr_primary_stem_only'] if webui_config['inference']['vr_primary_stem_only'] else False, interactive=True)
                vr_secondary_stem_only = gr.Checkbox(label=vr_secondary_stem_label, value=webui_config['inference']['vr_secondary_stem_only'] if webui_config['inference']['vr_secondary_stem_only'] else False, interactive=True)
            with gr.Tabs():
                with gr.TabItem(label=i18n("单个音频上传")):
                    vr_single_audio = gr.File(label="单个音频上传", type="filepath")
                with gr.TabItem(label=i18n("批量音频上传")):
                    with gr.Row():
                        vr_multiple_audio_input = gr.Textbox(label=i18n("输入目录"),value=webui_config['inference']['vr_multiple_audio_input'] if webui_config['inference']['vr_multiple_audio_input'] else "input/", interactive=True, scale=3)
                        vr_select_multi_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
                        vr_open_multi_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
            with gr.Row():
                vr_store_dir = gr.Textbox(label=i18n("输出目录"),value=webui_config['inference']['vr_store_dir'] if webui_config['inference']['vr_store_dir'] else "results/",interactive=True,scale=3)
                vr_select_store_btn = gr.Button(i18n("选择文件夹"), scale=1)
                vr_open_store_btn = gr.Button(i18n("打开文件夹"), scale=1)
            with gr.Accordion(i18n("以下是一些高级设置, 一般保持默认即可"), open=False):
                with gr.Row():
                    vr_batch_size = gr.Number(label=i18n("Batch Size: 一次要处理的批次数, 越大占用越多RAM, 处理速度加快"), minimum=1, value=webui_config['inference']['vr_batch_size'] if webui_config['inference']['vr_batch_size'] else 2, interactive=True)
                    vr_normalization = gr.Number(label=i18n("Normalization: 最大峰值振幅, 用于归一化输入和输出音频。取值为0-1"), minimum=0.0, maximum=1.0, step=0.01, value=webui_config['inference']['vr_normalization'] if webui_config['inference']['vr_normalization'] else 1, interactive=True)
                    vr_post_process_threshold = gr.Number(label=i18n("Post Process Threshold: 后处理特征阈值, 取值为0.1-0.3"), minimum=0.1, maximum=0.3, step=0.01, value=webui_config['inference']['vr_post_process_threshold'] if webui_config['inference']['vr_post_process_threshold'] else 0.2, interactive=True)
                with gr.Row():
                    vr_invert_spect = gr.Checkbox(label=i18n("Invert Spectrogram: 二级步骤将使用频谱图而非波形进行反转, 可能会提高质量, 但速度稍慢"), value=webui_config['inference']['vr_invert_spect'] if webui_config['inference']['vr_invert_spect'] else False, interactive=True)
                    vr_enable_tta = gr.Checkbox(label=i18n("Enable TTA: 启用“测试时增强”, 可能会提高质量, 但速度稍慢"), value=webui_config['inference']['vr_enable_tta'] if webui_config['inference']['vr_enable_tta'] else False, interactive=True)
                    vr_high_end_process = gr.Checkbox(label=i18n("High End Process: 将输出音频缺失的频率范围镜像输出"), value=webui_config['inference']['vr_high_end_process'] if webui_config['inference']['vr_high_end_process'] else False, interactive=True)
                    vr_enable_post_process = gr.Checkbox(label=i18n("Enable Post Process: 识别人声输出中残留的人工痕迹, 可改善某些歌曲的分离效果"), value=webui_config['inference']['vr_enable_post_process'] if webui_config['inference']['vr_enable_post_process'] else False, interactive=True)
                vr_debug_mode = gr.Checkbox(label=i18n("Debug Mode: 启用调试模式, 向开发人员反馈时, 请开启此模式"), value=webui_config['inference']['vr_debug_mode'] if webui_config['inference']['vr_debug_mode'] else False, interactive=True)
            with gr.Row():
                vr_start_single_inference = gr.Button(i18n("单个音频分离"), variant="primary")
                vr_start_multi_inference = gr.Button(i18n("批量音频分离"), variant="primary")
            with gr.Row():
                vr_output_message = gr.Textbox(label="Output Message", scale=4)
                stop_thread = gr.Button(i18n("强制停止"), scale=1)

            vr_select_model.change(fn=load_vr_model_stem,inputs=vr_select_model,outputs=[vr_primary_stem_only, vr_secondary_stem_only])
            vr_select_multi_input_dir.click(fn=select_folder, outputs=vr_multiple_audio_input)
            vr_open_multi_input_dir.click(fn=open_folder, inputs=vr_multiple_audio_input)
            vr_select_store_btn.click(fn=select_folder, outputs=vr_store_dir)
            vr_open_store_btn.click(fn=open_folder, inputs=vr_store_dir)
            vr_start_single_inference.click(fn=vr_inference_single,inputs=[vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_single_audio, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode],outputs=vr_output_message)
            vr_start_multi_inference.click(fn=vr_inference_multi,inputs=[vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode],outputs=vr_output_message)
            stop_thread.click(fn=stop_all_thread)

        with gr.TabItem(label=i18n("预设流程")):
            gr.Markdown(value=i18n("预设流程允许按照预设的顺序运行多个模型。每一个模型的输出将作为下一个模型的输入。"))
            with gr.Tabs():
                with gr.TabItem(label=i18n("使用预设")):
                    gr.Markdown(value=i18n("该模式下的UVR推理参数将直接沿用UVR分离页面的推理参数, 如需修改请前往UVR分离页面。<br>修改完成后, 还需要任意处理一首歌才能保存参数! "))
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(label=i18n("请选择预设"),choices=list(presets.keys()),value=webui_config['inference']['preset'] if webui_config['inference']['preset'] else None, interactive=True, scale=4)
                        output_format_flow = gr.Dropdown(label=i18n("输出格式"),choices=["wav", "mp3", "flac"],value=webui_config['inference']['output_format_flow'] if webui_config['inference']['output_format_flow'] else "wav", interactive=True, scale=1)
                    force_cpu = gr.Checkbox(label=i18n("使用CPU (注意: 使用CPU会导致速度非常慢) "),value=webui_config['inference']['force_cpu'] if webui_config['inference']['force_cpu'] else False,interactive=True)
                    with gr.Tabs():
                        with gr.TabItem(label=i18n("单个音频上传")):
                            single_audio_flow = gr.File(label=i18n("单个音频上传"), type="filepath")
                        with gr.TabItem(label=i18n("批量音频上传")):
                            with gr.Row():
                                input_folder_flow = gr.Textbox(label=i18n("输入目录"),value=webui_config['inference']['input_folder_flow'] if webui_config['inference']['input_folder_flow'] else "input/",interactive=True,scale=3)
                                select_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
                                open_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
                    with gr.Row():
                        store_dir_flow = gr.Textbox(label=i18n("输出目录"),value=webui_config['inference']['store_dir_flow'] if webui_config['inference']['store_dir_flow'] else "results/",interactive=True,scale=3)
                        select_output_dir = gr.Button(i18n("选择文件夹"), scale=1)
                        open_output_dir = gr.Button(i18n("打开文件夹"), scale=1)
                    with gr.Row():
                        single_inference_flow = gr.Button(i18n("单个音频分离"), variant="primary")
                        inference_flow = gr.Button(i18n("批量音频分离"), variant="primary")
                    with gr.Row():
                        output_message_flow = gr.Textbox(label="Output Message", scale=4)
                        stop_thread = gr.Button(i18n("强制停止"), scale=1)
                with gr.TabItem(label=i18n("制作预设")):
                    gr.Markdown(i18n("注意: MSST模型仅支持输出主要音轨, UVR模型支持自定义主要音轨输出。<br>同时输出次级音轨: 选择True将同时输出该次分离得到的次级音轨, **此音轨将直接保存至**输出目录下的secondary_output文件夹, **不会经过后续流程处理**<br>"))
                    preset_name_input = gr.Textbox(label=i18n("预设名称"), placeholder=i18n("请输入预设名称"), interactive=True)
                    with gr.Row():
                        model_type = gr.Dropdown(label=i18n("选择模型类型"), choices=MODEL_CHOICES, interactive=True)
                        model_name = gr.Dropdown(label=i18n("选择模型"), choices=[i18n("请先选择模型类型")], interactive=True, scale=2)
                        stem = gr.Dropdown(label=i18n("输出音轨"), choices=[i18n("请先选择模型")], interactive=True)
                        secondary_output = gr.Dropdown(label=i18n("同时输出次级音轨"), choices=["True", "False"], value="False", interactive=True)
                    add_to_flow = gr.Button(i18n("添加至流程"))
                    gr.Markdown(i18n("预设流程"))
                    preset_flow = gr.Dataframe(pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""], "secondary_output": [""]}), interactive=False, label=None)
                    reset_flow = gr.Button(i18n("重新输入"))
                    save_flow = gr.Button(i18n("保存上述预设流程"), variant="primary")
                    output_message_make = gr.Textbox(label="Output Message")
                with gr.TabItem(label=i18n("管理预设")):
                    gr.Markdown(i18n("此页面提供查看预设, 删除预设, 备份预设, 恢复预设等功能"))
                    preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=list(presets.keys()), interactive=True)
                    gr.Markdown(i18n("`model_type`: 模型类型；`model_name`: 模型名称；`stem`: 主要输出音轨；<br>`secondary_output`: 同时输出次级音轨。选择True将同时输出该次分离得到的次级音轨, **此音轨将直接保存至**输出目录下的secondary_output文件夹, **不会经过后续流程处理**"))
                    preset_flow_delete = gr.Dataframe(pd.DataFrame({"model_type": [i18n("请先选择预设")], "model_name": [i18n("请先选择预设")], "stem": [i18n("请先选择预设")], "secondary_output": [i18n("请先选择预设")]}), interactive=False, label=None)
                    delete_button = gr.Button(i18n("删除所选预设"), scale=1)
                    gr.Markdown(i18n("每次删除预设前, 将自动备份预设以免误操作。<br>你也可以点击“备份预设流程”按钮进行手动备份, 也可以从备份文件夹中恢复预设流程。"))
                    with gr.Row():
                        backup_preset = gr.Button(i18n("备份预设流程"))
                        open_preset_backup = gr.Button(i18n("打开备份文件夹"))
                    with gr.Row():
                        select_preset_backup = gr.Dropdown(label=i18n("选择需要恢复的预设流程备份"),choices=preset_backup_list(),interactive=True,scale=4)
                        restore_preset = gr.Button(i18n("恢复"), scale=1)
                    output_message_manage = gr.Textbox(label="Output Message")
                

            inference_flow.click(fn=run_inference_flow,inputs=[input_folder_flow, store_dir_flow, preset_dropdown, force_cpu, output_format_flow],outputs=output_message_flow)
            single_inference_flow.click(fn=run_single_inference_flow,inputs=[single_audio_flow, store_dir_flow, preset_dropdown, force_cpu, output_format_flow],outputs=output_message_flow)
            select_input_dir.click(fn=select_folder, outputs=input_folder_flow)
            open_input_dir.click(fn=open_folder, inputs=input_folder_flow)
            select_output_dir.click(fn=select_folder, outputs=store_dir_flow)
            open_output_dir.click(fn=open_folder, inputs=store_dir_flow)
            model_type.change(update_model_name, inputs=model_type, outputs=model_name)
            model_name.change(update_model_stem, inputs=[model_type, model_name], outputs=stem)
            add_to_flow.click(add_to_flow_func, [model_type, model_name, stem, secondary_output, preset_flow], preset_flow)
            save_flow.click(save_flow_func, [preset_name_input, preset_flow], [output_message_make, preset_name_delete, preset_dropdown])
            reset_flow.click(reset_flow_func, [], preset_flow)
            delete_button.click(delete_func, [preset_name_delete], [output_message_manage, preset_name_delete, preset_dropdown, preset_flow_delete, select_preset_backup])
            preset_name_delete.change(load_preset, inputs=preset_name_delete, outputs=preset_flow_delete)
            stop_thread.click(fn=stop_all_thread)
            restore_preset.click(fn=restore_preset_func,inputs=[select_preset_backup],outputs=[output_message_manage, preset_dropdown, preset_name_delete, preset_flow_delete])
            backup_preset.click(fn=backup_preset_func,outputs=[output_message_manage, select_preset_backup])
            open_preset_backup.click(open_folder, inputs=gr.Textbox(BACKUP, visible=False))

        with gr.TabItem(label=i18n("小工具")):
            with gr.Tabs():
                with gr.TabItem(label=i18n("音频格式转换")):
                    gr.Markdown(value=i18n("上传一个或多个音频文件并将其转换为指定格式。<br>支持的格式包括 .mp3, .flac, .wav, .ogg, .m4a, .wma, .aac...等等。<br>**不支持**网易云音乐/QQ音乐等加密格式, 如.ncm, .qmc等。"))
                    with gr.Row():
                        inputs = gr.Files(label=i18n("上传一个或多个音频文件"))
                        with gr.Column():
                            ffmpeg_output_format = gr.Dropdown(label=i18n("选择或输入音频输出格式"),choices=["wav", "flac", "mp3", "ogg", "m4a", "wma", "aac"],value=webui_config['tools']['ffmpeg_output_format'] if webui_config['tools']['ffmpeg_output_format'] else "wav",allow_custom_value=True,interactive=True)
                            ffmpeg_output_folder = gr.Textbox(label=i18n("选择音频输出目录"), value=webui_config['tools']['ffmpeg_output_folder'] if webui_config['tools']['ffmpeg_output_folder'] else "results/ffmpeg_output/", interactive=True)
                            with gr.Row():
                                select_ffmpeg_output_dir = gr.Button(i18n("选择文件夹"))
                                open_ffmpeg_output_dir = gr.Button(i18n("打开文件夹"))
                    convert_audio_button = gr.Button(i18n("转换音频"), variant="primary")
                    output_message_ffmpeg = gr.Textbox(label="Output Message")
                with gr.TabItem(label=i18n("合并音频")):
                    gr.Markdown(value=i18n("点击合并音频按钮后, 将自动把输入文件夹中的所有音频文件合并为一整个音频文件<br>目前支持的格式包括 .mp3, .flac, .wav, .ogg 这四种<br>合并后的音频会保存至输出目录中, 文件名为merged_audio.wav"))
                    with gr.Row():
                        merge_audio_input = gr.Textbox(label=i18n("输入目录"),value=webui_config['tools']['merge_audio_input'] if webui_config['tools']['merge_audio_input'] else "input/",interactive=True,scale=3)
                        select_merge_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
                        open_merge_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
                    with gr.Row():
                        merge_audio_output = gr.Textbox(label=i18n("输出目录"),value=webui_config['tools']['merge_audio_output'] if webui_config['tools']['merge_audio_output'] else "results/merge",interactive=True,scale=3)
                        select_merge_output_dir = gr.Button(i18n("选择文件夹"), scale=1)
                        open_merge_output_dir = gr.Button(i18n("打开文件夹"), scale=1)
                    merge_audio_button = gr.Button(i18n("合并音频"), variant="primary")
                    output_message_merge = gr.Textbox(label="Output Message")
                with gr.TabItem(label=i18n("计算SDR")):
                    with gr.Column():
                        gr.Markdown(value=i18n("上传两个**wav音频文件**并计算它们的[SDR](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021#evaluation-metric)。<br>SDR是一个用于评估模型质量的数值。数值越大, 模型算法结果越好。"))
                    with gr.Row():
                        true_audio = gr.File(label=i18n("原始音频"), type="filepath")
                        estimated_audio = gr.File(label=i18n("分离后的音频"), type="filepath")
                    compute_sdr_button = gr.Button(i18n("计算SDR"), variant="primary")
                    output_message_sdr = gr.Textbox(label="Output Message")
                with gr.TabItem(label = i18n("Ensemble模式")):
                    gr.Markdown(value = i18n("可用于集成不同算法的结果。具体的文档位于/docs/ensemble.md"))
                    with gr.Row():
                        files = gr.Files(label = i18n("上传多个音频文件"), type = "filepath", file_count = 'multiple')
                        with gr.Column():
                            with gr.Row():
                                ensemble_type = gr.Dropdown(choices = ["avg_wave", "median_wave", "min_wave", "max_wave", "avg_fft", "median_fft", "min_fft", "max_fft"],label = i18n("集成模式"),value = webui_config['tools']['ensemble_type'] if webui_config['tools']['ensemble_type'] else "avg_wave",interactive=True)
                                weights = gr.Textbox(label = i18n("权重(以空格分隔, 数量要与上传的音频一致)"), value = "1 1")
                            ensembl_output_path = gr.Textbox(label = i18n("输出目录"), value = webui_config['tools']['ensemble_output_folder'] if webui_config['tools']['ensemble_output_folder'] else "results/ensemble/",interactive=True)
                            with gr.Row():
                                select_ensembl_output_path = gr.Button(i18n("选择文件夹"))
                                open_ensembl_output_path = gr.Button(i18n("打开文件夹"))
                    ensemble_button = gr.Button(i18n("运行"), variant = "primary")
                    output_message_ensemble = gr.Textbox(label = "Output Message")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(i18n("### 集成模式"))
                            gr.Markdown(i18n("1. `avg_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的平均值<br>2. `median_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的中位数<br>3. `min_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的最小绝对值<br>4. `max_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的最大绝对值<br>5. `avg_fft`: 在频谱图 (短时傅里叶变换 (STFT) 2D变体) 上进行集成, 独立地找到频谱图的每个像素的平均值。平均后使用逆STFT得到原始的1D波形<br>6. `median_fft`: 与avg_fft相同, 但使用中位数代替平均值 (仅在集成3个或更多来源时有用) <br>7. `min_fft`: 与avg_fft相同, 但使用最小函数代替平均值 (减少激进程度) <br>8. `max_fft`: 与avg_fft相同, 但使用最大函数代替平均值 (增加激进程度) "))
                        with gr.Column():
                            gr.Markdown(i18n("### 注意事项"))
                            gr.Markdown(i18n("1. min_fft可用于进行更保守的合成, 它将减少更激进模型的影响。<br>2. 最好合成等质量的模型。在这种情况下, 它将带来增益。如果其中一个模型质量不好, 它将降低整体质量。<br>3. 在原仓库作者的实验中, 与其他方法相比, avg_wave在SDR分数上总是更好或相等。<br>4. 上传的文件名**不能包含空格**, 最终会在输出目录下生成一个`ensemble_<集成模式>.wav`。"))
                with gr.TabItem(label=i18n("歌声转MIDI")):
                    gr.Markdown(value=i18n("歌声转MIDI功能使用开源项目[SOME](https://github.com/openvpi/SOME/), 可以将分离得到的**干净的歌声**转换成.mid文件。<br>【必须】若想要使用此功能, 请先下载权重文件[model_steps_64000_simplified.ckpt](https://hf-mirror.com/Sucial/SOME_Models/resolve/main/model_steps_64000_simplified.ckpt)并将其放置在程序目录下的`tools/SOME_weights`文件夹内。文件命名不可随意更改! <br>【重要】只能上传wav格式的音频! "))
                    with gr.Row():
                        some_input_audio = gr.File(label=i18n("上传wav格式音频"), type="filepath")
                        with gr.Column():
                            audio_bpm = gr.Number(label=i18n("输入音频BPM"), value=120, interactive=True)
                            some_output_folder = gr.Textbox(label=i18n("输出目录"),value=webui_config['tools']['some_output_folder'] if webui_config['tools']['some_output_folder'] else "results/some/",interactive=True,scale=3)
                            with gr.Row():
                                select_some_output_dir = gr.Button(i18n("选择文件夹"))
                                open_some_output_dir = gr.Button(i18n("打开文件夹"))
                    some_button = gr.Button(i18n("开始转换"), variant="primary")
                    output_message_some = gr.Textbox(label="Output Message")
                    gr.Markdown(i18n("### 注意事项"))
                    gr.Markdown(i18n("1. 音频BPM (每分钟节拍数) 可以通过MixMeister BPM Analyzer等软件测量获取。<br>2. 为保证MIDI提取质量, 音频文件请采用干净清晰无混响底噪人声。<br>3. 输出MIDI不带歌词信息, 需要用户自行添加歌词。<br>4. 实际使用体验中部分音符会出现断开的现象, 需自行修正。SOME的模型主要面向DiffSinger唱法模型自动标注, 比正常用户在创作中需要的MIDI更加精细, 因而可能导致模型倾向于对音符进行切分。<br>5. 提取的MIDI没有量化/没有对齐节拍/不适配BPM, 需自行到各编辑器中手动调整。"))

            convert_audio_button.click(fn=convert_audio, inputs=[inputs, ffmpeg_output_format, ffmpeg_output_folder], outputs=output_message_ffmpeg)
            select_ffmpeg_output_dir.click(fn=select_folder, outputs=ffmpeg_output_folder)
            open_ffmpeg_output_dir.click(fn=open_folder, inputs=ffmpeg_output_folder)
            merge_audio_button.click(merge_audios, [merge_audio_input, merge_audio_output], outputs=output_message_merge)
            select_merge_input_dir.click(fn=select_folder, outputs=merge_audio_input)
            open_merge_input_dir.click(fn=open_folder, inputs=merge_audio_input)
            select_merge_output_dir.click(fn=select_folder, outputs=merge_audio_output)
            open_merge_output_dir.click(fn=open_folder, inputs=merge_audio_output)
            compute_sdr_button.click(process_audio, [true_audio, estimated_audio], outputs=output_message_sdr)
            ensemble_button.click(fn = ensemble, inputs = [files, ensemble_type, weights, ensembl_output_path],outputs = output_message_ensemble)
            select_ensembl_output_path.click(fn = select_folder, outputs = ensembl_output_path)
            open_ensembl_output_path.click(fn = open_folder, inputs = ensembl_output_path)
            select_some_output_dir.click(fn=select_folder, outputs=some_output_folder)
            open_some_output_dir.click(fn=open_folder, inputs=some_output_folder)
            some_button.click(fn=some_inference,inputs=[some_input_audio, audio_bpm, some_output_folder],outputs=output_message_some)

        with gr.TabItem(label=i18n("安装模型")):
            with gr.Tabs():
                with gr.TabItem(label=i18n("下载官方模型")):
                    uvr_model_folder = webui_config['settings']['uvr_model_dir']
                    gr.Markdown(value=i18n("若自动下载出现报错或下载过慢, 请点击手动下载, 跳转至下载链接。手动下载完成后, 请根据你选择的模型类型放置到对应文件夹内。"))
                    gr.Markdown(value=i18n("### 当前UVR模型目录: ") + f"`{uvr_model_folder}`" + i18n(", 如需更改, 请前往设置页面。"))
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Row():
                                model_type_dropdown = gr.Dropdown(label=i18n("选择模型类型"), choices=MODEL_CHOICES, scale=1)
                                download_model_name_dropdown = gr.Dropdown(label=i18n("选择模型"), choices=[i18n("请先选择模型类型")], scale=3)
                            with gr.Row():
                                open_model_dir = gr.Button(i18n("打开MSST模型目录"))
                                open_uvr_model_dir = gr.Button(i18n("打开UVR模型目录"))
                            with gr.Row():
                                download_button = gr.Button(i18n("自动下载"), variant="primary")
                                manual_download_button = gr.Button(i18n("手动下载"), variant="primary")
                            output_message_download = gr.Textbox(label="Output Message")
                            restart_webui = gr.Button(i18n("重启WebUI"), variant="primary")
                        with gr.Column(scale=1):
                            gr.Markdown(i18n("### 注意事项"))
                            gr.Markdown(value=i18n("1. MSST模型默认下载在pretrain/<模型类型>文件夹下。UVR模型默认下载在设置中的UVR模型目录中。<br>2. 下加载进度可以打开终端查看。如果一直卡着不动或者速度很慢, 在确信网络正常的情况下请尝试重启WebUI。<br>3. 若下载失败, 会在模型目录**留下一个损坏的模型**, 请**务必**打开模型目录手动删除! <br>4. 点击“重启WebUI”按钮后, 会短暂性的失去连接, 随后会自动开启一个新网页。"))
                            gr.Markdown(i18n("### 模型下载链接"))
                            gr.Markdown(i18n("1. 自动从Github, Huggingface或镜像站下载模型。<br>2. 你也可以在此整合包下载链接中的All_Models文件夹中找到所有可用的模型并下载。"))
                            gr.Markdown(value=i18n("### 模型安装完成后, 需重启WebUI刷新模型列表"))
                with gr.TabItem(label=i18n("安装非官方MSST模型")):
                    gr.Markdown(value=i18n("你可以从其他途径获取非官方MSST模型, 在此页面完成配置文件设置后, 即可正常使用。<br>注意: 仅支持'.ckpt', '.th', '.chpt'格式的模型。模型显示名字为模型文件名。<br>选择模型类型: 共有三个可选项。依次代表人声相关模型, 多音轨分离模型, 单音轨分离模型。仅用于区分模型大致类型, 可任意选择。<br>选择模型类别: 此选项关系到模型是否能正常推理使用, 必须准确选择!"))
                    with gr.Row():
                        unmsst_model = gr.File(label=i18n("上传非官方MSST模型"), type="filepath")
                        unmsst_config = gr.File(label=i18n("上传非官方MSST模型配置文件"), type="filepath")
                    with gr.Row():
                        unmodel_class = gr.Dropdown(label=i18n("选择模型类型"), choices=["vocal_models", "multi_stem_models", "single_stem_models"], interactive=True)
                        unmodel_type = gr.Dropdown(label=i18n("选择模型类别"), choices=MODEL_TYPE, interactive=True)
                        unmsst_model_link = gr.Textbox(label=i18n("模型下载链接 (非必须，若无，可跳过)"), value="", interactive=True, scale=2)
                    unmsst_model_install = gr.Button(i18n("安装非官方MSST模型"), variant="primary")
                    output_message_unmsst = gr.Textbox(label="Output Message")
                with gr.TabItem(label=i18n("安装非官方VR模型")):
                    gr.Markdown(value=i18n("你可以从其他途径获取非官方UVR模型, 在此页面完成配置文件设置后, 即可正常使用。<br>注意: 仅支持'.pth'格式的模型。模型显示名字为模型文件名。"))
                    with gr.Row():
                        unvr_model = gr.File(label=i18n("上传非官方VR模型"), type="filepath")
                        with gr.Column():
                            with gr.Row():
                                unvr_primary_stem = gr.Textbox(label=i18n("主要音轨名称"), value="", interactive=True)
                                unvr_secondary_stem = gr.Textbox(label=i18n("次要音轨名称"), value="", interactive=True)
                            model_param = gr.Dropdown(label=i18n("选择模型参数"), choices=get_all_model_param(), interactive=True)
                            with gr.Row():
                                is_karaoke_model = gr.Checkbox(label=i18n("是否为Karaoke模型"), value=False, interactive=True)
                                is_BV_model = gr.Checkbox(label=i18n("是否为BV模型"), value=False, interactive=True)
                                is_VR51_model = gr.Checkbox(label=i18n("是否为VR 5.1模型"), value=False, interactive=True)
                    balance_value = gr.Number(label="balance_value", value=0.0, minimum=0.0, maximum=0.9, step=0.1, interactive=True, visible=False)
                    with gr.Row():
                        out_channels = gr.Number(label="Out Channels", value=32, minimum=1, step=1, interactive=True, visible=False)
                        out_channels_lstm = gr.Number(label="Out Channels (LSTM layer)", value=128, minimum=1, step=1, interactive=True, visible=False)
                    upload_param = gr.File(label=i18n("上传参数文件"), type="filepath", interactive=True, visible=False)
                    unvr_model_link = gr.Textbox(label=i18n("模型下载链接 (非必须，若无，可跳过)"), value="", interactive=True)
                    unvr_model_install = gr.Button(i18n("安装非官方VR模型"), variant="primary")
                    output_message_unvr = gr.Textbox(label="Output Message")

            model_type_dropdown.change(fn=upgrade_download_model_name,inputs=[model_type_dropdown],outputs=[download_model_name_dropdown])
            download_button.click(fn=download_model,inputs=[model_type_dropdown, download_model_name_dropdown],outputs=output_message_download)
            manual_download_button.click(fn=manual_download_model,inputs=[model_type_dropdown, download_model_name_dropdown],outputs=output_message_download)
            open_model_dir.click(open_folder, inputs=gr.Textbox(MODEL_FOLDER, visible=False))
            open_uvr_model_dir.click(open_folder, inputs=gr.Textbox(uvr_model_folder, visible=False))
            restart_webui.click(webui_restart)
            is_BV_model.change(fn=update_vr_param, inputs=[is_BV_model, is_VR51_model, model_param], outputs=[balance_value, out_channels, out_channels_lstm, upload_param])
            is_VR51_model.change(fn=update_vr_param, inputs=[is_BV_model, is_VR51_model, model_param], outputs=[balance_value, out_channels, out_channels_lstm, upload_param])
            model_param.change(fn=update_vr_param, inputs=[is_BV_model, is_VR51_model, model_param], outputs=[balance_value, out_channels, out_channels_lstm, upload_param])
            unmsst_model_install.click(fn=install_unmsst_model, inputs=[unmsst_model, unmsst_config, unmodel_class, unmodel_type, unmsst_model_link], outputs=output_message_unmsst)
            unvr_model_install.click(fn=install_unvr_model, inputs=[unvr_model, unvr_primary_stem, unvr_secondary_stem, model_param, is_karaoke_model, is_BV_model, is_VR51_model, balance_value, out_channels, out_channels_lstm, upload_param, unvr_model_link], outputs=output_message_unvr)

        with gr.TabItem(label=i18n("MSST训练")):
            gr.Markdown(value=i18n("此页面提供数据集制作教程, 训练参数选择, 以及一键训练。有关配置文件的修改和数据集文件夹的详细说明请参考MSST原项目: [https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)<br>在开始下方的模型训练之前, 请先进行训练数据的制作。<br>说明: 数据集类型即训练集制作Step 1中你选择的类型, 1: Type1; 2: Type2; 3: Type3; 4: Type4, 必须与你的数据集类型相匹配。"))
            with gr.Tabs():
                with gr.TabItem(label=i18n("训练")):
                    with gr.Row():
                        train_model_type = gr.Dropdown(label=i18n("选择训练模型类型"),choices=MODEL_TYPE,value=webui_config['training']['model_type'] if webui_config['training']['model_type'] else None,interactive=True,scale=1)
                        train_config_path = gr.Textbox(label=i18n("配置文件路径"),value=webui_config['training']['config_path'] if webui_config['training']['config_path'] else i18n("请输入配置文件路径或选择配置文件"),interactive=True,scale=3)
                        select_train_config_path = gr.Button(i18n("选择配置文件"), scale=1)
                    with gr.Row():
                        train_dataset_type = gr.Dropdown(label=i18n("数据集类型"),choices=[1, 2, 3, 4],value=webui_config['training']['dataset_type'] if webui_config['training']['dataset_type'] else None,interactive=True,scale=1)
                        train_dataset_path = gr.Textbox(label=i18n("数据集路径"),value=webui_config['training']['dataset_path'] if webui_config['training']['dataset_path'] else i18n("请输入或选择数据集文件夹"),interactive=True,scale=3)
                        select_train_dataset_path = gr.Button(i18n("选择数据集文件夹"), scale=1)
                    with gr.Row():
                        train_valid_path = gr.Textbox(label=i18n("验证集路径"),value=webui_config['training']['valid_path'] if webui_config['training']['valid_path'] else i18n("请输入或选择验证集文件夹"),interactive=True,scale=4)
                        select_train_valid_path = gr.Button(i18n("选择验证集文件夹"), scale=1)
                    with gr.Row():
                        train_num_workers = gr.Number(label=i18n("num_workers: 数据集读取线程数, 0为自动"),value=webui_config['training']['num_workers'] if webui_config['training']['num_workers'] else 0,interactive=True,minimum=0,maximum=cpu_count(),step=1)
                        train_device_ids = gr.Textbox(label=i18n("device_ids: 选择显卡, 多卡用户请使用空格分隔"),value=webui_config['training']['device_ids'] if webui_config['training']['device_ids'] else "0",interactive=True)
                        train_seed = gr.Number(label=i18n("随机数种子, 0为随机"), value="0")
                    with gr.Row():
                        train_pin_memory = gr.Checkbox(label=i18n("是否将加载的数据放置在固定内存中, 默认为否"), value=webui_config['training']['pin_memory'], interactive=True)
                        train_accelerate = gr.Checkbox(label=i18n("是否使用加速训练, 对于多显卡用户会加快训练"), value=webui_config['training']['accelerate'], interactive=True)
                        train_pre_validate = gr.Checkbox(label=i18n("是否在训练前验证模型, 仅对加速训练有效"), value=webui_config['training']['pre_valid'], interactive=True)
                    with gr.Row():
                        train_use_multistft_loss = gr.Checkbox(label=i18n("是否使用MultiSTFT Loss, 默认为否"), value=webui_config['training']['use_multistft_loss'], interactive=True)
                        train_use_mse_loss = gr.Checkbox(label=i18n("是否使用MSE loss, 默认为否"), value=webui_config['training']['use_mse_loss'], interactive=True)
                        train_use_l1_loss = gr.Checkbox(label=i18n("是否使用L1 loss, 默认为否"), value=webui_config['training']['use_l1_loss'], interactive=True)
                    with gr.Row():
                        train_results_path = gr.Textbox(label=i18n("模型保存路径"),value=webui_config['training']['results_path'] if webui_config['training']['results_path'] else i18n("请输入或选择模型保存文件夹"),interactive=True,scale=3)
                        select_train_results_path = gr.Button(i18n("选择文件夹"), scale=1)
                        open_train_results_path = gr.Button(i18n("打开文件夹"), scale=1)
                    with gr.Row():
                        train_start_check_point = gr.Dropdown(label=i18n("初始模型: 继续训练或微调模型训练时, 请选择初始模型, 否则将从头开始训练! "), choices=["None"], value="None", interactive=True, scale=4)
                        reflesh_start_check_point = gr.Button(i18n("刷新初始模型列表"), scale=1)
                    save_train_config = gr.Button(i18n("保存上述训练配置"))
                    start_train_button = gr.Button(i18n("开始训练"), variant="primary")
                    gr.Markdown(value=i18n("点击开始训练后, 请到终端查看训练进度或报错, 下方不会输出报错信息, 想要停止训练可以直接关闭终端。在训练过程中, 你也可以关闭网页, 仅**保留终端**。"))
                    with gr.Row():
                        output_message_train = gr.Textbox(label="Output Message", scale=4)
                        stop_thread = gr.Button(i18n("强制停止"), scale=1)

                    select_train_config_path.click(fn=select_yaml_file, outputs=train_config_path)
                    select_train_dataset_path.click(fn=select_folder, outputs=train_dataset_path)
                    select_train_valid_path.click(fn=select_folder, outputs=train_valid_path)
                    select_train_results_path.click(fn=select_folder, outputs=train_results_path)
                    open_train_results_path.click(fn=open_folder, inputs=train_results_path)
                    save_train_config.click(fn=save_training_config,inputs=[train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers,train_device_ids, train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_accelerate, train_pre_validate],outputs=output_message_train)
                    start_train_button.click(fn=start_training,inputs=[train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers, train_device_ids,train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_start_check_point, train_accelerate, train_pre_validate],outputs=output_message_train)
                    reflesh_start_check_point.click(fn=update_train_start_check_point,inputs=train_results_path,outputs=train_start_check_point)
                    stop_thread.click(fn=stop_all_thread)

                with gr.TabItem(label=i18n("验证")):
                    gr.Markdown(value=i18n("此页面用于手动验证模型效果, 测试验证集, 输出SDR测试信息。输出的信息会存放在输出文件夹的results.txt中。<br>下方参数将自动加载训练页面的参数, 在训练页面点击保存训练参数后, 重启WebUI即可自动加载。当然你也可以手动输入参数。<br>"))
                    with gr.Row():
                        valid_model_type = gr.Dropdown(label=i18n("选择模型类型"),choices=MODEL_TYPE,value=webui_config['training']['model_type'] if webui_config['training']['model_type'] else None,interactive=True,scale=1)
                        valid_config_path = gr.Textbox(label=i18n("配置文件路径"),value=webui_config['training']['config_path'] if webui_config['training']['config_path'] else i18n("请输入配置文件路径或选择配置文件"),interactive=True,scale=3)
                        select_valid_config_path = gr.Button(i18n("选择配置文件"), scale=1)
                    with gr.Row():
                        valid_model_path = gr.Textbox(label=i18n("模型路径"),value=i18n("请输入或选择模型文件"),interactive=True,scale=4)
                        select_valid_model_path = gr.Button(i18n("选择模型文件"), scale=1)
                    with gr.Row():
                        valid_path = gr.Textbox(label=i18n("验证集路径"),value=webui_config['training']['valid_path'] if webui_config['training']['valid_path'] else i18n("请输入或选择验证集文件夹"),interactive=True,scale=4)
                        select_valid_path = gr.Button(i18n("选择验证集文件夹"), scale=1)
                    with gr.Row():
                        valid_results_path = gr.Textbox(label=i18n("输出目录"),value="results/",interactive=True,scale=3)
                        select_valid_results_path = gr.Button(i18n("选择文件夹"), scale=1)
                        open_valid_results_path = gr.Button(i18n("打开文件夹"), scale=1)
                    with gr.Row():
                        valid_device_ids = gr.Textbox(label=i18n("选择显卡, 多卡用户请使用空格分隔GPU ID"),value=webui_config['training']['device_ids'] if webui_config['training']['device_ids'] else "0",interactive=True)
                        valid_num_workers = gr.Number(label=i18n("验证集读取线程数, 0为自动"),value=webui_config['training']['num_workers'] if webui_config['training']['num_workers'] else 0,interactive=True,minimum=0,maximum=cpu_count(),step=1)
                        valid_extension = gr.Dropdown(label=i18n("选择验证集音频格式"),choices=["wav", "flac", "mp3"],value="wav",interactive=True,allow_custom_value=True)
                    valid_pin_memory = gr.Checkbox(label=i18n("是否将加载的数据放置在固定内存中, 默认为否"), value=webui_config['training']['pin_memory'], interactive=True)
                    valid_button = gr.Button(i18n("开始验证"), variant="primary")
                    with gr.Row():
                        valid_output_message = gr.Textbox(label="Output Message", scale=4)
                        stop_thread = gr.Button(i18n("强制停止"), scale=1)

                    select_valid_config_path.click(fn=select_yaml_file, outputs=valid_config_path)
                    select_valid_model_path.click(fn=select_file, outputs=valid_model_path)
                    select_valid_path.click(fn=select_folder, outputs=valid_path)
                    select_valid_results_path.click(fn=select_folder, outputs=valid_results_path)
                    open_valid_results_path.click(fn=open_folder, inputs=valid_results_path)
                    valid_button.click(fn=validate_model,inputs=[valid_model_type, valid_config_path, valid_model_path, valid_path, valid_results_path, valid_device_ids, valid_num_workers, valid_extension, valid_pin_memory],outputs=valid_output_message)
                    stop_thread.click(fn=stop_all_thread)

                with gr.TabItem(label=i18n("训练集制作指南")):
                    with gr.Accordion(i18n("Step 1: 数据集制作"), open=False):
                        gr.Markdown(value=i18n("请**任选下面四种类型之一**制作数据集文件夹, 并按照给出的目录层级放置你的训练数据。完成后, 记录你的数据集**文件夹路径**以及你选择的**数据集类型**, 以便后续使用。"))
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("# Type 1 (MUSDB)")
                                gr.Markdown(i18n("不同的文件夹。每个文件夹包含所需的所有stems, 格式为stem_name.wav。与MUSDBHQ18数据集相同。在最新的代码版本中, 可以使用flac替代wav。<br>例如: "))
                                gr.Markdown("""
                                    your_datasets_folder<br>
                                    ├───Song 1<br>
                                    │   ├───vocals.wav<br>
                                    │   ├───bass.wav<br>
                                    │   ├───drums.wav<br>
                                    │   └───other.wav<br>
                                    ├───Song 2<br>
                                    │   ├───vocals.wav<br>
                                    │   ├───bass.wav<br>
                                    │   ├───drums.wav<br>
                                    │   └───other.wav<br>
                                    ├───Song 3<br>
                                        └───...<br>
                                    """)
                            with gr.Column():
                                gr.Markdown("# Type 2 (Stems)")
                                gr.Markdown(i18n("每个文件夹是stem_name。文件夹中包含仅由所需stem组成的wav文件。<br>例如: "))
                                gr.Markdown("""
                                    your_datasets_folder<br>
                                    ├───vocals<br>
                                    │   ├───vocals_1.wav<br>
                                    │   ├───vocals_2.wav<br>
                                    │   ├───vocals_3.wav<br>
                                    │   └───...<br>
                                    ├───bass<br>
                                    │   ├───bass_1.wav<br>
                                    │   ├───bass_2.wav<br>
                                    │   ├───bass_3.wav<br>
                                    │   └───...<br>
                                    ├───drums<br>
                                        └───...<br>
                                    """)
                            with gr.Column():
                                gr.Markdown("# Type 3 (CSV file)")
                                gr.Markdown(i18n("可以提供以下结构的CSV文件 (或CSV文件列表) <br>例如: "))
                                gr.Markdown("""
                                    instrum,path<br>
                                    vocals,/path/to/dataset/vocals_1.wav<br>
                                    vocals,/path/to/dataset2/vocals_v2.wav<br>
                                    vocals,/path/to/dataset3/vocals_some.wav<br>
                                    ...<br>
                                    drums,/path/to/dataset/drums_good.wav<br>
                                    ...<br>
                                    """)
                            with gr.Column():
                                gr.Markdown("# Type 4 (MUSDB Aligned)")
                                gr.Markdown(i18n("与类型1相同, 但在训练过程中所有乐器都将来自歌曲的相同位置。<br>例如: "))
                                gr.Markdown("""
                                    your_datasets_folder<br>
                                    ├───Song 1<br>
                                    │   ├───vocals.wav<br>
                                    │   ├───bass.wav<br>
                                    │   ├───drums.wav<br>
                                    │   └───other.wav<br>
                                    ├───Song 2<br>
                                    │   ├───vocals.wav<br>
                                    │   ├───bass.wav<br>
                                    │   ├───drums.wav<br>
                                    │   └───other.wav<br>
                                    ├───Song 3<br>
                                        └───...<br>
                                    """)
                    with gr.Accordion(i18n("Step 2: 验证集制作"), open=False):
                        gr.Markdown(value=i18n("验证集制作。验证数据集**必须**与上面数据集制作的Type 1(MUSDB)数据集**结构相同** (**无论你使用哪种类型的数据集进行训练**) , 此外每个文件夹还必须包含每首歌的mixture.wav, mixture.wav是所有stem的总和<br>例如: "))
                        gr.Markdown("""
                            your_datasets_folder<br>
                            ├───Song 1<br>
                            │   ├───vocals.wav<br>
                            │   ├───bass.wav<br>
                            │   ├───drums.wav<br>
                            │   ├───other.wav<br>
                            │   └───mixture.wav<br>
                            ├───Song 2<br>
                            │   ├───vocals.wav<br>
                            │   ├───bass.wav<br>
                            │   ├───drums.wav<br>
                            │   ├───other.wav<br>
                            │   └───mixture.wav<br>
                            ├───Song 3<br>
                                └───...<br>
                            """)
                    with gr.Accordion(i18n("Step 3: 选择并修改修改配置文件"), open=False):
                        gr.Markdown(value=i18n("请先明确你想要训练的模型类型, 然后选择对应的配置文件进行修改。<br>目前有以下几种模型类型: ") + str(MODEL_TYPE) + i18n("<br>确定好模型类型后, 你可以前往整合包根目录中的configs_backup文件夹下找到对应的配置文件模板。复制一份模板, 然后根据你的需求进行修改。修改完成后记下你的配置文件路径, 以便后续使用。<br>特别说明: config_musdb18_xxx.yaml是针对MUSDB18数据集的配置文件。<br>"))
                        open_config_template = gr.Button(
                            i18n("打开配置文件模板文件夹"), variant="primary")
                        open_config_template.click(open_folder, inputs=gr.Textbox("configs_backup", visible=False))
                        gr.Markdown(value=i18n("你可以使用下表根据你的GPU选择用于训练的BS_Roformer模型的batch_size参数。表中提供的批量大小值适用于单个GPU。如果你有多个GPU, 则需要将该值乘以GPU的数量。"))
                        roformer_data = {
                            "chunk_size": [131584, 131584, 131584, 131584, 131584, 131584, 263168, 263168, 352800, 352800, 352800, 352800],
                            "dim": [128, 256, 384, 512, 256, 256, 128, 256, 128, 256, 384, 512],
                            "depth": [6, 6, 6, 6, 8, 12, 6, 6, 6, 6, 12, 12],
                            "batch_size (A6000 48GB)": [10, 8, 7, 6, 6, 4, 4, 3, 2, 2, 1, '-'],
                            "batch_size (3090/4090 24GB)": [5, 4, 3, 3, 3, 2, 2, 1, 1, 1, '-', '-'],
                            "batch_size (16GB)": [3, 2, 2, 2, 2, 1, 1, 1, '-', '-', '-', '-']
                        }
                        gr.DataFrame(pd.DataFrame(roformer_data))
                    with gr.Accordion(i18n("Step 4: 数据增强"), open=False):
                        gr.Markdown(value=i18n("数据增强可以动态更改stem, 通过从旧样本创建新样本来增加数据集的大小。现在, 数据增强的控制在配置文件中进行。下面是一个包含所有可用数据增强的完整配置示例。你可以将其复制到你的配置文件中以使用数据增强。<br>注意:<br>1. 要完全禁用所有数据增强, 可以从配置文件中删除augmentations部分或将enable设置为false。<br>2. 如果要禁用某些数据增强, 只需将其设置为0。<br>3. all部分中的数据增强应用于所有stem。<br>4. vocals/bass等部分中的数据增强仅应用于相应的stem。你可以为training.instruments中给出的所有stem创建这样的部分。"))
                        augmentations_config = load_augmentations_config()
                        gr.Code(value=augmentations_config, language="yaml")

        with gr.TabItem(label=i18n("设置")):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        gpu_list = gr.Textbox(label=i18n("GPU信息"), value=get_device(), interactive=False)
                        plantform_info = gr.Textbox(label=i18n("系统信息"), value=get_platform(), interactive=False)
                    with gr.Row():
                        set_webui_port = gr.Number(label=i18n("设置WebUI端口, 0为自动"), value=webui_config["settings"].get("port", 0), interactive=True)
                        set_language = gr.Dropdown(label=i18n("选择语言"), choices=language_dict.keys(), value=get_language(), interactive=True)
                        set_download_link = gr.Dropdown(label=i18n("选择MSST模型下载链接"), choices=["Auto", i18n("huggingface.co (需要魔法)"), i18n("hf-mirror.com (镜像站可直连)")], value=webui_config['settings']['download_link'] if webui_config['settings']['download_link'] else "Auto", interactive=True)
                    with gr.Row():
                        open_local_link = gr.Checkbox(label=i18n("对本地局域网开放WebUI: 开启后, 同一局域网内的设备可通过'本机IP:端口'的方式访问WebUI。"), value=webui_config['settings']['local_link'], interactive=True)
                        open_share_link = gr.Checkbox(label=i18n("开启公共链接: 开启后, 他人可通过公共链接访问WebUI。链接有效时长为72小时。"), value=webui_config['settings']['share_link'], interactive=True)
                    with gr.Row():
                        select_uvr_model_dir = gr.Textbox(label=i18n("选择UVR模型目录"),value=webui_config['settings']['uvr_model_dir'] if webui_config['settings']['uvr_model_dir'] else "pretrain/VR_Models",interactive=True,scale=4)
                        select_uvr_model_dir_button = gr.Button(i18n("选择文件夹"), scale=1)
                    with gr.Row():
                        update_message = gr.Textbox(label=i18n("检查更新"), value=i18n("当前版本: ") + PACKAGE_VERSION + i18n(", 请点击检查更新按钮"), interactive=False,scale=3)
                        check_update = gr.Button(i18n("检查更新"), scale=1)
                        goto_github = gr.Button(i18n("前往Github瞅一眼"))
                    with gr.Row():
                        reset_all_webui_config = gr.Button(i18n("重置WebUI路径记录"), variant="primary")
                        reset_seetings = gr.Button(i18n("重置WebUI设置"), variant="primary")
                    restart_webui = gr.Button(i18n("重启WebUI"), variant="primary")
                    setting_output_message = gr.Textbox(label="Output Message")
                with gr.Column(scale=1):
                    gr.Markdown(i18n("### 设置说明"))
                    gr.Markdown(i18n("### 选择UVR模型目录"))
                    gr.Markdown(i18n("如果你的电脑中有安装UVR5, 你不必重新下载一遍UVR5模型, 只需在下方“选择UVR模型目录”中选择你的UVR5模型目录, 定位到models/VR_Models文件夹。<br>例如: E:/Program Files/Ultimate Vocal Remover/models/VR_Models 点击保存设置或重置设置后, 需要重启WebUI以更新。"))
                    gr.Markdown(i18n("### 检查更新"))
                    gr.Markdown(i18n("从Github检查更新, 需要一定的网络要求。点击检查更新按钮后, 会自动检查是否有最新版本。你可以前往此整合包的下载链接或访问Github仓库下载最新版本。"))
                    gr.Markdown(i18n("### 重置WebUI路径记录"))
                    gr.Markdown(i18n("将所有输入输出目录重置为默认路径, 预设/模型/配置文件以及上面的设置等**不会重置**, 无需担心。重置WebUI设置后, 需要重启WebUI。"))
                    gr.Markdown(i18n("### 重置WebUI设置"))
                    gr.Markdown(i18n("仅重置WebUI设置, 例如UVR模型路径, WebUI端口等。重置WebUI设置后, 需要重启WebUI。"))
                    gr.Markdown(i18n("### 重启WebUI"))
                    gr.Markdown(i18n("点击 “重启WebUI” 按钮后, 会短暂性的失去连接, 随后会自动开启一个新网页。"))

            restart_webui.click(fn=webui_restart, outputs=setting_output_message)
            check_update.click(fn=check_webui_update, outputs=update_message)
            goto_github.click(fn=webui_goto_github)
            select_uvr_model_dir_button.click(fn=select_folder, outputs=select_uvr_model_dir)
            reset_seetings.click(fn=reset_settings,inputs=[],outputs=setting_output_message)
            reset_all_webui_config.click(fn=reset_webui_config,outputs=setting_output_message)
            set_webui_port.change(fn=save_port_to_config,inputs=[set_webui_port],outputs=setting_output_message)
            select_uvr_model_dir.change(fn=save_uvr_modeldir,inputs=[select_uvr_model_dir],outputs=setting_output_message)
            set_language.change(fn=change_language,inputs=[set_language],outputs=setting_output_message)
            set_download_link.change(fn=change_download_link,inputs=[set_download_link],outputs=setting_output_message)
            open_share_link.change(fn=change_share_link,inputs=[open_share_link],outputs=setting_output_message)
            open_local_link.change(fn=change_local_link,inputs=[open_local_link],outputs=setting_output_message)

local_link = "0.0.0.0" if webui_config["settings"].get("local_link", False) else None
server_port = None if webui_config["settings"].get("port", 0) == 0 else webui_config["settings"].get("port", 0)
isShare = webui_config["settings"].get("share_link", False)
app.launch(inbrowser=True, server_name=local_link, server_port=server_port, share=isShare)
