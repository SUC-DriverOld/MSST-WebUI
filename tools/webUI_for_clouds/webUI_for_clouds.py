# This webUI file is for MSST-WebUI-Clouds

import gradio as gr
import subprocess
import os
import sys
import re
import time
import tempfile
import shutil
import json
import requests
import yaml
import librosa
import numpy as np
import pandas as pd
import platform
import warnings
import locale
import threading
import psutil
from datetime import datetime
from ml_collections import ConfigDict
from mir_eval.separation import bss_eval_sources
from pydub import AudioSegment
from rich.console import Console
from torch import cuda, backends
from multiprocessing import cpu_count
from .download_models import download_model


PACKAGE_VERSION = "1.5"
PRESETS = "data/preset_data.json"
BACKUP = "backup/"
MODELS = "data/model_map.json"
WEBUI_CONFIG = "data/webui_config.json"
VR_MODEL = "data/vr_model.json"
MODEL_FOLDER = "pretrain/"
CONFIG_TEMPLATE_FOLDER = "configs_template/"
VERSION_CONFIG = "data/version.json"
TEMP_PATH = "temp"
MODEL_TYPE = ['bs_roformer', 'mel_band_roformer', 'segm_models', 'htdemucs', 'mdx23c', 'swin_upernet', 'bandit']
FFMPEG = "ffmpeg"
PYTHON = sys.executable

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

def setup_webui():
    print(f"Music-Source-Separation-Training-Inference-Webui v{PACKAGE_VERSION} For-Clouds")
    print(i18n("[INFO] 设备信息：") + str(get_device()))


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


def load_msst_model():
    config = load_configs(MODELS)
    model_list = []
    for keys in config.keys():
        for model in config[keys]:
            model_list.append(model["name"])
    return model_list


def get_msst_model(model_name):
    config = load_configs(MODELS)
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
                download_link = download_link.replace("huggingface.co", main_link)
                return model_path, config_path, model_type, download_link
    raise gr.Error(i18n("模型不存在!"))


def load_vr_model():
    config = load_configs(VR_MODEL)
    return config.keys()


def load_vr_model_stem(model):
    config = load_configs(VR_MODEL)
    for keys in config.keys():
        if keys == model:
            primary_stem = config[keys]["primary_stem"]
            secondary_stem = config[keys]["secondary_stem"]
            vr_primary_stem_only = gr.Checkbox(label=f"{primary_stem} Only", value=False, interactive=True)
            vr_secondary_stem_only = gr.Checkbox(label=f"{secondary_stem} Only", value=False, interactive=True)
            return vr_primary_stem_only, vr_secondary_stem_only
    raise gr.Error(i18n("模型不存在!"))


def load_presets_list():
    config = load_configs(PRESETS)
    if config == {}:
        return [i18n("无预设")]
    return list(config.keys())


def save_training_config(train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers, train_device_ids, train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_accelerate):
    try:
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
        config['training']['results_path'] = train_results_path
        save_configs(config, WEBUI_CONFIG)
        return i18n("配置保存成功!")
    except Exception as e:
        print(e)
        return i18n("配置保存失败!")


def reset_webui_config():
    try:
        config = load_configs(WEBUI_CONFIG)
        for key in config['training_backup']:
            config['training'][key] = config['training_backup'][key]
        for key in config['inference_backup']:
            config['inference'][key] = config['inference_backup'][key]
        for key in config['tools_backup']:
            config['tools'][key] = config['tools_backup'][key]
        
        save_configs(config, WEBUI_CONFIG)
        return i18n("记录重置成功, 请重启WebUI刷新! ")
    except Exception as e:
        print(e)
        return i18n("记录重置失败!")


def init_selected_vr_model():
    webui_config = load_configs(WEBUI_CONFIG)
    config = load_configs(VR_MODEL)
    model = webui_config['inference']['vr_select_model']
    if not model:
        vr_primary_stem_only = i18n("仅输出主音轨")
        vr_secondary_stem_only = i18n("仅输出次音轨")
        return vr_primary_stem_only, vr_secondary_stem_only
    for keys in config.keys():
        if keys == model:
            primary_stem = config[keys]["primary_stem"]
            secondary_stem = config[keys]["secondary_stem"]
            vr_primary_stem_only = f"{primary_stem} Only"
            vr_secondary_stem_only = f"{secondary_stem} Only"
            return vr_primary_stem_only, vr_secondary_stem_only
    vr_primary_stem_only = i18n("仅输出主音轨")
    vr_secondary_stem_only = i18n("仅输出次音轨")
    return vr_primary_stem_only, vr_secondary_stem_only


def update_train_start_check_point(path):
    if not os.path.isdir(path):
        raise gr.Error(i18n("请先选择模型保存路径! "))
    ckpt_files = [f for f in os.listdir(path) if f.endswith(('.ckpt', '.pth', '.th'))]
    return gr.Dropdown(label=i18n("初始模型"), choices=ckpt_files if ckpt_files else ["None"])


def update_inference_settings(selected_model):
    _, config_path, _, _ = get_msst_model(selected_model)
    config = load_configs(config_path)
    if config.inference.get('batch_size'):
        batch_size = gr.Textbox(label="batch_size", value=str(
            config.inference.get('batch_size')), interactive=True)
    else:
        batch_size = gr.Textbox(label="batch_size", value=i18n("该模型不支持修改此值"), interactive=False)
    if config.inference.get('dim_t'):
        dim_t = gr.Textbox(label="dim_t", value=str(
            config.inference.get('dim_t')), interactive=True)
    else:
            dim_t = gr.Textbox(label="dim_t", value=i18n("该模型不支持修改此值"), interactive=False)
    if config.inference.get('num_overlap'):
        num_overlap = gr.Textbox(label="num_overlap", value=str(
            config.inference.get('num_overlap')), interactive=True)
    else:
        num_overlap = gr.Textbox(label="num_overlap", value=i18n("该模型不支持修改此值"), interactive=False)
    if config.inference.get('normalize'):
        normalize = gr.Checkbox(label="normalize", value=config.inference.get('normalize'), interactive=True)
    else:
        normalize = gr.Checkbox(label=i18n("normalize (该模型不支持修改此值) "), value=False, interactive=False)
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
            raise gr.Error(i18n("发生错误! 请前往终端查看详细信息"))
    except Exception as e:
        print(e)
        raise gr.Error(i18n("发生错误! 请前往终端查看详细信息"))


def stop_all_thread():
    global stop_all_threads
    for thread in threading.enumerate():
        if thread.name in ["msst_inference", "vr_inference", "msst_training", "msst_valid"]:
            stop_all_threads = True
            gr.Info(i18n("已停止进程"))


def run_multi_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, force_cpu):
    if not bool(re.match(r'^(\d+)(?:\s(?!\1)\d+)*$', gpu_id)):
        raise gr.Error(i18n("GPU ID格式错误, 请重新输入"))
    if selected_model == "":
        raise gr.Error(i18n("请选择模型"))
    if input_folder == "":
        raise gr.Error(i18n("请选择输入目录"))
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    if download_model("msst", selected_model):
        run_inference(selected_model, input_folder, store_dir,extract_instrumental, gpu_id, force_cpu)
        return i18n("处理完成! 分离完成的音频文件已保存在") + store_dir


def run_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, force_cpu, extra_store_dir=None):
    if extra_store_dir and not os.path.exists(extra_store_dir):
        os.makedirs(extra_store_dir)
    start_check_point, config_path, model_type, _ = get_msst_model(selected_model)
    gpu_ids = gpu_id if not force_cpu else "0"
    extract_instrumental_option = "--extract_instrumental" if extract_instrumental else ""
    force_cpu_option = "--force_cpu" if force_cpu else ""
    extra_store_dir = f"--extra_store_dir \"{extra_store_dir}\"" if extra_store_dir else ""
    command = f"{PYTHON} msst_inference.py --model_type {model_type} --config_path \"{config_path}\" --start_check_point \"{start_check_point}\" --input_folder \"{input_folder}\" --store_dir \"{store_dir}\" --device_ids {gpu_ids} {extract_instrumental_option} {force_cpu_option} {extra_store_dir}"
    msst_inference = threading.Thread(target=run_command, args=(command,), name="msst_inference")
    msst_inference.start()
    msst_inference.join()


def vr_inference_multi(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
    if not os.path.isdir(vr_multiple_audio_input):
        return i18n("请选择输入文件夹")
    if not vr_select_model:
        return i18n("请选择模型")
    if not vr_store_dir:
        return i18n("请选择输出目录")
    if not os.path.exists(vr_store_dir):
        os.makedirs(vr_store_dir)
    if download_model("uvr", vr_select_model):
        vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
        return i18n("处理完成, 结果已保存至") + vr_store_dir


def vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode, save_another_stem=False, extra_output_dir=None):
    config = load_configs(WEBUI_CONFIG)
    model_file_dir = config['settings']['uvr_model_dir']
    model_mapping = load_configs(VR_MODEL)
    primary_stem = model_mapping[vr_select_model]["primary_stem"]
    secondary_stem = model_mapping[vr_select_model]["secondary_stem"]
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
    sample_rate = 44100
    vr_batch_size = int(vr_batch_size)
    vr_aggression = int(vr_aggression)
    use_cpu = "--use_cpu" if vr_use_cpu else ""
    vr_enable_tta = "--vr_enable_tta" if vr_enable_tta else ""
    vr_high_end_process = "--vr_high_end_process" if vr_high_end_process else ""
    vr_enable_post_process = "--vr_enable_post_process" if vr_enable_post_process else ""
    save_another_stem = "--save_another_stem" if save_another_stem else ""
    extra_output_dir = f"--extra_output_dir \"{extra_output_dir}\"" if extra_output_dir else ""
    command = f"{PYTHON} uvr_inference.py \"{audio_file}\" {debug_mode} --model_filename \"{model_filename}\" --output_format {output_format} --output_dir \"{output_dir}\" --model_file_dir \"{model_file_dir}\" {invert_spect} --normalization {normalization} {single_stem} --sample_rate {sample_rate} {use_cpu} --vr_batch_size {vr_batch_size} --vr_window_size {vr_window_size} --vr_aggression {vr_aggression} {vr_enable_tta} {vr_high_end_process} {vr_enable_post_process} --vr_post_process_threshold {vr_post_process_threshold} {save_another_stem} {extra_output_dir}"
    vr_inference = threading.Thread(target=run_command, args=(command,), name="vr_inference")
    vr_inference.start()
    vr_inference.join()


def update_model_name(model_type):
    if model_type == "UVR_VR_Models":
        model_map = load_vr_model()
        return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)
    else:
        model_map = load_msst_model()
        return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)


def update_model_stem(model_type, model_name):
    if model_type == "UVR_VR_Models":
        config = load_configs(VR_MODEL)
        for keys in config.keys():
            if keys == model_name:
                primary_stem = config[keys]["primary_stem"]
                secondary_stem = config[keys]["secondary_stem"]
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


def run_inference_flow(input_folder, store_dir, preset_name, force_cpu):
    start_time = time.time()
    preset_data = load_configs(PRESETS)
    if not preset_name in preset_data.keys():
        return i18n("预设") + preset_name + i18n("不存在")
    config = load_configs(WEBUI_CONFIG)
    model_list = preset_data[preset_name]
    input_to_use = input_folder
    tmp_store_dir = tempfile.mkdtemp()
    i = 0
    for step in model_list.keys():
        if i == 0:
            input_to_use = input_folder
        elif i < len(model_list.keys()) - 1 and i > 0:
            if input_to_use != input_folder:
                shutil.rmtree(input_to_use)
            input_to_use = tmp_store_dir
            tmp_store_dir = tempfile.mkdtemp()
        elif i == len(model_list.keys()) - 1:
            input_to_use = tmp_store_dir
            tmp_store_dir = store_dir
        console = Console()
        model_name = model_list[step]["model_name"]
        console.rule(f"[yellow]Step {i+1}: Running inference using {model_name}", style="yellow")
        if model_list[step]["model_type"] == "MSST_Models":
            gpu_id = config['inference']['gpu_id'] if not force_cpu else "0"
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
            if download_model("msst", model_name):
                run_inference(model_name, input_to_use, tmp_store_dir, extract_instrumental, gpu_id, force_cpu, extra_store_dir)
        elif model_list[step]["model_type"] == "UVR_VR_Models":
            vr_model_config = load_configs(VR_MODEL)
            stem = model_list[step]["stem"]
            vr_select_model = model_name
            vr_window_size = config['inference']['vr_window_size']
            vr_aggression = config['inference']['vr_aggression']
            vr_output_format = "wav"
            vr_use_cpu = force_cpu
            vr_primary_stem_only = True if stem == vr_model_config[model_name]["primary_stem"] else False
            vr_secondary_stem_only = True if stem == vr_model_config[model_name]["secondary_stem"] else False
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
            if download_model("uvr", vr_select_model):
                vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode, save_another_stem, extra_output_dir)
        i += 1
    if tmp_store_dir != store_dir:
        for file_name in os.listdir(tmp_store_dir):
            shutil.move(os.path.join(tmp_store_dir, file_name),
                        os.path.join(store_dir, file_name))
        shutil.rmtree(tmp_store_dir)
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


def process_audio(true_path, estimated_path):
    true_audio, _ = librosa.load(true_path, sr=44100, mono=False)
    if true_audio.ndim == 1:
        true_audio = np.vstack((true_audio, true_audio))
    elif true_audio.ndim == 2 and true_audio.shape[0] == 1:
        true_audio = np.vstack((true_audio[0], true_audio[0]))
    estimated_audio, _ = librosa.load(estimated_path, sr=44100, mono=False)
    if estimated_audio.ndim == 1:
        estimated_audio = np.vstack((estimated_audio, estimated_audio))
    elif estimated_audio.ndim == 2 and estimated_audio.shape[0] == 1:
        estimated_audio = np.vstack((estimated_audio[0], estimated_audio[0]))
    min_length = min(true_audio.shape[1], estimated_audio.shape[1])
    true_audio = true_audio[:, :min_length]
    estimated_audio = estimated_audio[:, :min_length]
    sdr, _, _, _ = bss_eval_sources(true_audio, estimated_audio)
    print(f"SDR: {sdr}")
    return sdr


def ensemble(files, ensemble_mode, weights, output_path):
    if len(files) < 2:
        return i18n("请上传至少2个文件")
    if len(files) != len(weights.split()):
        return i18n("上传的文件数目与权重数目不匹配")
    else:
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


def start_training(train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers, train_device_ids, train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_start_check_point, train_accelerate):
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
    command = f"{PYTHON} {train_file} --model_type {model_type} --config_path \"{config_path}\" {start_check_point} --results_path \"{results_path}\" --data_path \"{data_path}\" --dataset_type {dataset_type} --valid_path \"{valid_path}\" --num_workers {num_workers} --device_ids {device_ids} --seed {seed} --pin_memory {pin_memory} {use_multistft_loss} {use_mse_loss} {use_l1_loss}"
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


setup_webui()
with gr.Blocks(
        theme=gr.Theme.load('tools/themes/theme_schema@1.2.2.json')
) as app:
    gr.Markdown(value=f"""### Music-Source-Separation-Training-Inference-Webui v{PACKAGE_VERSION} For-Clouds""")
    gr.Markdown(value=i18n("仅供个人娱乐和非商业用途, 禁止用于血腥/暴力/性相关/政治相关内容。[点击前往教程文档](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)<br>本整合包完全免费, 严禁以任何形式倒卖, 如果你从任何地方**付费**购买了本整合包, 请**立即退款**。<br> 整合包作者: [bilibili@阿狸不吃隼舞](https://space.bilibili.com/403335715) [Github@KitsuneX07](https://github.com/KitsuneX07) | [Bilibili@Sucial](https://space.bilibili.com/445022409) [Github@SUC-DriverOld](https://github.com/SUC-DriverOld) | Gradio主题: [Gradio Theme](https://huggingface.co/spaces/NoCrypt/miku)"))
    with gr.Tabs():
        webui_config = load_configs(WEBUI_CONFIG)
        presets = load_configs(PRESETS)
        models = load_configs(MODELS)
        vr_model = load_configs(VR_MODEL)

        with gr.TabItem(label=i18n("MSST分离")):
            gr.Markdown(value=i18n("MSST音频分离原项目地址: [https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)"))
            selected_model = gr.Dropdown(label=i18n("选择模型"), choices=load_msst_model(), value=None, interactive=True)
            with gr.Row():
                gpu_id = gr.Textbox(label=i18n("选择使用的GPU ID, 多卡用户请使用空格分隔GPU ID。可前往设置页面查看显卡信息。"), value="0",interactive=True)
                with gr.Column():
                    force_cpu = gr.Checkbox(label=i18n("使用CPU (注意: 使用CPU会导致速度非常慢) "), value=False, interactive=True)
                    extract_instrumental = gr.Checkbox(label=i18n("同时输出次级音轨"), value=False, interactive=True)
            multiple_audio_input = gr.Textbox(label=i18n("输入目录"), value="input/", interactive=True)
            store_dir = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=True)
            with gr.Accordion(i18n("推理参数设置 (一般不需要动) "), open=False):
                gr.Markdown(value=i18n("只有在点击保存后才会生效。参数直接写入配置文件, 无法撤销。假如不知道如何设置, 请保持默认值。<br>请牢记自己修改前的参数数值, 防止出现问题以后无法恢复。请确保输入正确的参数, 否则可能会导致模型无法正常运行。<br>假如修改后无法恢复, 请点击``重置``按钮, 这会使得配置文件恢复到默认值。"))
                with gr.Row():
                    batch_size = gr.Textbox(label=i18n("batch_size: 批次大小, 一般不需要改"), value=i18n("请先选择模型"))
                    dim_t = gr.Textbox(label=i18n("dim_t: 时序维度大小, 一般不需要改 (部分模型没有此参数)"), value=i18n("请先选择模型"))
                    num_overlap = gr.Textbox(label=i18n("num_overlap: 窗口重叠长度, 数值越小速度越快, 但会牺牲效果"), value=i18n("请先选择模型"))
                normalize = gr.Checkbox(label=i18n("normalize: 是否对音频进行归一化处理 (部分模型没有此参数)"), value=False, interactive=False)
                reset_config_button = gr.Button(i18n("重置配置"), variant="secondary")
                save_config_button = gr.Button(i18n("保存配置"), variant="primary")
            inference_multiple = gr.Button(i18n("批量音频分离"), variant="primary")
            with gr.Row():
                output_message = gr.Textbox(label="Output Message", scale=4)
                stop_thread = gr.Button(i18n("强制停止"), scale=1)

            inference_multiple.click(fn=run_multi_inference, inputs=[selected_model, multiple_audio_input, store_dir, extract_instrumental, gpu_id, force_cpu],outputs=output_message)
            selected_model.change(fn=update_inference_settings,inputs=[selected_model],outputs=[batch_size, dim_t, num_overlap, normalize])
            save_config_button.click(fn=save_config,inputs=[selected_model, batch_size, dim_t, num_overlap, normalize],outputs=output_message)
            reset_config_button.click(fn=reset_config,inputs=[selected_model],outputs=output_message)
            stop_thread.click(fn=stop_all_thread)

        with gr.TabItem(label=i18n("UVR分离")):
            gr.Markdown(value=i18n("说明: 本整合包仅融合了UVR的VR Architecture模型, MDX23C和HtDemucs类模型可以直接使用前面的MSST音频分离。<br>使用UVR模型进行音频分离时, 若有可用的GPU, 软件将自动选择, 否则将使用CPU进行分离。<br>UVR分离使用项目: [https://github.com/nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 并进行了优化。"))
            vr_select_model = gr.Dropdown(label=i18n("选择模型"), choices=load_vr_model(), value=webui_config['inference']['vr_select_model'] if webui_config['inference']['vr_select_model'] else None,interactive=True)
            with gr.Row():
                vr_window_size = gr.Dropdown(label=i18n("Window Size: 窗口大小, 用于平衡速度和质量"), choices=["320", "512", "1024"], value="512", interactive=True)
                vr_aggression = gr.Number(label=i18n("Aggression: 主干提取强度, 范围-100-100, 人声请选5"), minimum=-100, maximum=100, value=5, interactive=True)
                vr_output_format = gr.Dropdown(label=i18n("输出格式"), choices=["wav", "flac", "mp3"], value="wav", interactive=True)
            with gr.Row():
                vr_primary_stem_label, vr_secondary_stem_label = init_selected_vr_model()
                vr_use_cpu = gr.Checkbox(label=i18n("使用CPU"), value=False, interactive=True)
                vr_primary_stem_only = gr.Checkbox(label=vr_primary_stem_label, value=False, interactive=True)
                vr_secondary_stem_only = gr.Checkbox(label=vr_secondary_stem_label, value=False, interactive=True)
            vr_multiple_audio_input = gr.Textbox(label=i18n("输入目录"),value="input/",interactive=True)
            vr_store_dir = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=True)
            with gr.Accordion(i18n("以下是一些高级设置, 一般保持默认即可"), open=False):
                with gr.Row():
                    vr_batch_size = gr.Number(label=i18n("Batch Size: 一次要处理的批次数, 越大占用越多RAM, 处理速度加快"), minimum=1, value=2, interactive=True)
                    vr_normalization = gr.Number(label=i18n("Normalization: 最大峰值振幅, 用于归一化输入和输出音频。取值为0-1"), minimum=0.0, maximum=1.0, step=0.01, value=1, interactive=True)
                    vr_post_process_threshold = gr.Number(label=i18n("Post Process Threshold: 后处理特征阈值, 取值为0.1-0.3"), minimum=0.1, maximum=0.3, step=0.01, value=0.2, interactive=True)
                with gr.Row():
                    vr_invert_spect = gr.Checkbox(label=i18n("Invert Spectrogram: 二级步骤将使用频谱图而非波形进行反转, 可能会提高质量, 但速度稍慢"), value=False, interactive=True)
                    vr_enable_tta = gr.Checkbox(label=i18n("Enable TTA: 启用“测试时间增强”, 可能会提高质量, 但速度稍慢"), value=False, interactive=True)
                    vr_high_end_process = gr.Checkbox(label=i18n("High End Process: 将输出音频缺失的频率范围镜像输出"), value=False, interactive=True)
                    vr_enable_post_process = gr.Checkbox(label=i18n("Enable Post Process: 识别人声输出中残留的人工痕迹, 可改善某些歌曲的分离效果"), value=False, interactive=True)
                vr_debug_mode = gr.Checkbox(label=i18n("Debug Mode: 启用调试模式, 向开发人员反馈时, 请开启此模式"), value=False, interactive=True)
            vr_start_multi_inference = gr.Button(i18n("批量音频分离"), variant="primary")
            with gr.Row():
                vr_output_message = gr.Textbox(label="Output Message", scale=4)
                stop_thread = gr.Button(i18n("强制停止"), scale=1)

            vr_select_model.change(fn=load_vr_model_stem,inputs=vr_select_model,outputs=[vr_primary_stem_only, vr_secondary_stem_only])
            vr_start_multi_inference.click(fn=vr_inference_multi,inputs=[vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode],outputs=vr_output_message)
            stop_thread.click(fn=stop_all_thread)

        with gr.TabItem(label=i18n("预设流程")):
            gr.Markdown(value=i18n("预设流程允许按照预设的顺序运行多个模型。每一个模型的输出将作为下一个模型的输入。"))
            with gr.Tabs():
                with gr.TabItem(label=i18n("使用预设")):
                    preset_dropdown = gr.Dropdown(label=i18n("请选择预设"),choices=list(presets.keys()),value=None,interactive=True)
                    force_cpu = gr.Checkbox(label=i18n("使用CPU (注意: 使用CPU会导致速度非常慢) "), value=False, interactive=True)
                    input_folder_flow = gr.Textbox(label=i18n("输入目录"), value="input/", interactive=True)
                    store_dir_flow = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=True)
                    inference_flow = gr.Button(i18n("批量音频分离"), variant="primary")
                    with gr.Row():
                        output_message_flow = gr.Textbox(label="Output Message", scale=4)
                        stop_thread = gr.Button(i18n("强制停止"), scale=1)
                with gr.TabItem(label=i18n("制作预设")):
                    gr.Markdown(i18n("注意: MSST模型仅支持输出主要音轨, UVR模型支持自定义主要音轨输出。<br>同时输出次级音轨: 选择True将同时输出该次分离得到的次级音轨, **此音轨将直接保存至**输出目录下的secondary_output文件夹, **不会经过后续流程处理**<br>"))
                    preset_name_input = gr.Textbox(label=i18n("预设名称"), placeholder=i18n("请输入预设名称"), interactive=True)
                    with gr.Row():
                        model_type = gr.Dropdown(label=i18n("选择模型类型"), choices=["MSST_Models", "UVR_VR_Models"], interactive=True)
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
                    preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=load_presets_list(), interactive=True)
                    gr.Markdown(i18n("`model_type`: 模型类型；`model_name`: 模型名称；`stem`: 主要输出音轨；<br>`secondary_output`: 同时输出次级音轨。选择True将同时输出该次分离得到的次级音轨, **此音轨将直接保存至**输出目录下的secondary_output文件夹, **不会经过后续流程处理**"))
                    preset_flow_delete = gr.Dataframe(pd.DataFrame({"model_type": [i18n("请先选择预设")], "model_name": [i18n("请先选择预设")], "stem": [i18n("请先选择预设")], "secondary_output": [i18n("请先选择预设")]}), interactive=False, label=None)
                    delete_button = gr.Button(i18n("删除所选预设"), scale=1)
                    gr.Markdown(i18n("每次删除预设前, 将自动备份预设以免误操作。<br>你也可以点击“备份预设流程”按钮进行手动备份, 也可以从备份文件夹中恢复预设流程。"))
                    backup_preset = gr.Button(i18n("备份预设流程"))
                    with gr.Row():
                        select_preset_backup = gr.Dropdown(label=i18n("选择需要恢复的预设流程备份"),choices=preset_backup_list(),interactive=True,scale=4)
                        restore_preset = gr.Button(i18n("恢复"), scale=1)
                    output_message_manage = gr.Textbox(label="Output Message")
                

            inference_flow.click(fn=run_inference_flow,inputs=[input_folder_flow, store_dir_flow, preset_dropdown, force_cpu],outputs=output_message_flow)
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

        with gr.TabItem(label=i18n("小工具")):
            with gr.Tabs():
                with gr.TabItem(label=i18n("音频格式转换")):
                    gr.Markdown(value=i18n("上传一个或多个音频文件并将其转换为指定格式。<br>支持的格式包括 .mp3, .flac, .wav, .ogg, .m4a, .wma, .aac...等等。<br>**不支持**网易云音乐/QQ音乐等加密格式, 如.ncm, .qmc等。"))
                    with gr.Row():
                        inputs = gr.Textbox(label=i18n("上传一个或多个音频文件"), value="input/", interactive=True, scale=4)
                        ffmpeg_output_format = gr.Dropdown(label=i18n("选择或输入音频输出格式"), choices=["wav", "flac", "mp3", "ogg", "m4a", "wma", "aac"], value="wav", allow_custom_value=True, interactive=True, scale=2)
                    ffmpeg_output_folder = gr.Textbox(label=i18n("选择音频输出目录"), value="results/ffmpeg_output/", interactive=True)
                    convert_audio_button = gr.Button(i18n("转换音频"), variant="primary")
                    output_message_ffmpeg = gr.Textbox(label="Output Message")
                with gr.TabItem(label=i18n("合并音频")):
                    gr.Markdown(value=i18n("点击合并音频按钮后, 将自动把输入文件夹中的所有音频文件合并为一整个音频文件<br>目前支持的格式包括 .mp3, .flac, .wav, .ogg 这四种<br>合并后的音频会保存至输出目录中, 文件名为merged_audio.wav"))
                    merge_audio_input = gr.Textbox(label=i18n("输入目录"),value="input/",interactive=True,scale=3)
                    merge_audio_output = gr.Textbox(label=i18n("输出目录"),value="results/merge",interactive=True,scale=3)
                    merge_audio_button = gr.Button(i18n("合并音频"), variant="primary")
                    output_message_merge = gr.Textbox(label="Output Message")
                with gr.TabItem(label=i18n("计算SDR")):
                    gr.Markdown(value=i18n("上传两个**wav音频文件**并计算它们的[SDR](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021#evaluation-metric)。<br>SDR是一个用于评估模型质量的数值。数值越大, 模型算法结果越好。"))
                    with gr.Row():
                        true_audio = gr.Textbox(label=i18n("原始音频"), value="input/no_audio.wav", interactive=True)
                        estimated_audio = gr.Textbox(label=i18n("分离后的音频"), value="input/no_audio.wav", interactive=True)
                    compute_sdr_button = gr.Button(i18n("计算SDR"), variant="primary")
                    output_message_sdr = gr.Textbox(label="Output Message")
                with gr.TabItem(label = i18n("Ensemble模式")):
                    gr.Markdown(value = i18n("可用于集成不同算法的结果。具体的文档位于/docs/ensemble.md"))
                    files = gr.Textbox(label=i18n("上传多个音频文件"), value = "input/", interactive=True)
                    with gr.Row():
                        ensemble_type = gr.Dropdown(choices = ["avg_wave", "median_wave", "min_wave", "max_wave", "avg_fft", "median_fft", "min_fft", "max_fft"],label = i18n("集成模式"), value="avg_wave", interactive=True)
                        weights = gr.Textbox(label = i18n("权重(以空格分隔, 数量要与上传的音频一致)"), value="1 1")
                    ensembl_output_path = gr.Textbox(label = i18n("输出目录"), value="results/ensemble/", interactive=True)
                    ensemble_button = gr.Button(i18n("运行"), variant="primary")
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
                    some_input_audio = gr.Textbox(label=i18n("上传wav格式音频"), value="input/no_audio.wav", interactive=True)
                    audio_bpm = gr.Number(label=i18n("输入音频BPM"), value=120, interactive=True)
                    some_output_folder = gr.Textbox(label=i18n("输出目录"),value="results/some/", interactive=True)
                    some_button = gr.Button(i18n("开始转换"), variant="primary")
                    output_message_some = gr.Textbox(label="Output Message")
                    gr.Markdown(i18n("### 注意事项"))
                    gr.Markdown(i18n("1. 音频BPM (每分钟节拍数) 可以通过MixMeister BPM Analyzer等软件测量获取。<br>2. 为保证MIDI提取质量, 音频文件请采用干净清晰无混响底噪人声。<br>3. 输出MIDI不带歌词信息, 需要用户自行添加歌词。<br>4. 实际使用体验中部分音符会出现断开的现象, 需自行修正。SOME的模型主要面向DiffSinger唱法模型自动标注, 比正常用户在创作中需要的MIDI更加精细, 因而可能导致模型倾向于对音符进行切分。<br>5. 提取的MIDI没有量化/没有对齐节拍/不适配BPM, 需自行到各编辑器中手动调整。"))

            convert_audio_button.click(fn=convert_audio, inputs=[inputs, ffmpeg_output_format, ffmpeg_output_folder], outputs=output_message_ffmpeg)
            merge_audio_button.click(merge_audios, [merge_audio_input, merge_audio_output], outputs=output_message_merge)
            compute_sdr_button.click(process_audio, [true_audio, estimated_audio], outputs=output_message_sdr)
            ensemble_button.click(fn = ensemble, inputs = [files, ensemble_type, weights, ensembl_output_path],outputs = output_message_ensemble)
            some_button.click(fn=some_inference,inputs=[some_input_audio, audio_bpm, some_output_folder],outputs=output_message_some)

        with gr.TabItem(label=i18n("MSST训练")):
            gr.Markdown(value=i18n("此页面提供数据集制作教程, 训练参数选择, 以及一键训练。有关配置文件的修改和数据集文件夹的详细说明请参考MSST原项目: [https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)<br>在开始下方的模型训练之前, 请先进行训练数据的制作。"))
            with gr.Tabs():
                with gr.TabItem(label=i18n("训练")):
                    with gr.Row():
                        train_model_type = gr.Dropdown(label=i18n("选择训练模型类型"),choices=MODEL_TYPE,value=webui_config['training']['model_type'] if webui_config['training']['model_type'] else None,interactive=True,scale=1)
                        train_config_path = gr.Textbox(label=i18n("配置文件路径"),value=webui_config['training']['config_path'] if webui_config['training']['config_path'] else i18n("请输入配置文件路径或选择配置文件"),interactive=True,scale=3)
                    with gr.Row():
                        train_dataset_type = gr.Dropdown(label=i18n("数据集类型"),choices=[1, 2, 3, 4],value=webui_config['training']['dataset_type'] if webui_config['training']['dataset_type'] else None,interactive=True,scale=1)
                        train_dataset_path = gr.Textbox(label=i18n("数据集路径"),value=webui_config['training']['dataset_path'] if webui_config['training']['dataset_path'] else i18n("请输入或选择数据集文件夹"),interactive=True,scale=3)
                    gr.Markdown(value=i18n("说明: 数据集类型即训练集制作Step 1中你选择的类型, 1: Type1; 2: Type2; 3: Type3; 4: Type4, 必须与你的数据集类型相匹配。"))
                    train_valid_path = gr.Textbox(label=i18n("验证集路径"),value=webui_config['training']['valid_path'] if webui_config['training']['valid_path'] else i18n("请输入或选择验证集文件夹"),interactive=True,scale=4)
                    with gr.Row():
                        train_num_workers = gr.Number(label=i18n("num_workers: 数据集读取线程数, 0为自动"),value=webui_config['training']['num_workers'] if webui_config['training']['num_workers'] else 0,interactive=True,minimum=0,maximum=cpu_count(),step=1)
                        train_device_ids = gr.Textbox(label=i18n("device_ids: 选择显卡, 多卡用户请使用空格分隔"),value=webui_config['training']['device_ids'] if webui_config['training']['device_ids'] else "0",interactive=True)
                        train_seed = gr.Number(label=i18n("随机数种子, 0为随机"), value="0")
                    with gr.Row():
                        train_pin_memory = gr.Checkbox(label=i18n("是否将加载的数据放置在固定内存中, 默认为否"), value=webui_config['training']['pin_memory'], interactive=True)
                        train_use_multistft_loss = gr.Checkbox(label=i18n("是否使用MultiSTFT Loss, 默认为否"), value=webui_config['training']['use_multistft_loss'], interactive=True)
                        train_use_mse_loss = gr.Checkbox(label=i18n("是否使用MSE loss, 默认为否"), value=webui_config['training']['use_mse_loss'], interactive=True)
                        train_use_l1_loss = gr.Checkbox(label=i18n("是否使用L1 loss, 默认为否"), value=webui_config['training']['use_l1_loss'], interactive=True)
                        train_accelerate = gr.Checkbox(label=i18n("(实验中) 是否使用加速训练, 对于多显卡用户会加快训练"), value=False, interactive=True)
                    with gr.Row():
                        train_results_path = gr.Textbox(label=i18n("模型保存路径"),value=webui_config['training']['results_path'] if webui_config['training']['results_path'] else i18n("请输入或选择模型保存文件夹"),interactive=True,scale=3)
                    with gr.Row():
                        train_start_check_point = gr.Dropdown(label=i18n("初始模型: 继续训练或微调模型训练时, 请选择初始模型, 否则将从头开始训练! "), choices=["None"], value="None", interactive=True, scale=4)
                        reflesh_start_check_point = gr.Button(i18n("刷新初始模型列表"), scale=1)
                    save_train_config = gr.Button(i18n("保存上述训练配置"))
                    start_train_button = gr.Button(i18n("开始训练"), variant="primary")
                    gr.Markdown(value=i18n("点击开始训练后, 请到终端查看训练进度或报错, 下方不会输出报错信息, 想要停止训练可以直接关闭终端。在训练过程中, 你也可以关闭网页, 仅**保留终端**。"))
                    with gr.Row():
                        output_message_train = gr.Textbox(label="Output Message", scale=4)
                        stop_thread = gr.Button(i18n("强制停止"), scale=1)

                    save_train_config.click(fn=save_training_config,inputs=[train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers,train_device_ids, train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_accelerate],outputs=output_message_train)
                    start_train_button.click(fn=start_training,inputs=[train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers, train_device_ids,train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_start_check_point, train_accelerate],outputs=output_message_train)
                    reflesh_start_check_point.click(fn=update_train_start_check_point,inputs=train_results_path,outputs=train_start_check_point)
                    stop_thread.click(fn=stop_all_thread)

                with gr.TabItem(label=i18n("验证")):
                    gr.Markdown(value=i18n("此页面用于手动验证模型效果, 测试验证集, 输出SDR测试信息。输出的信息会存放在输出文件夹的results.txt中。<br>下方参数将自动加载训练页面的参数, 在训练页面点击保存训练参数后, 重启WebUI即可自动加载。当然你也可以手动输入参数。<br>"))
                    with gr.Row():
                        valid_model_type = gr.Dropdown(label=i18n("选择模型类型"),choices=MODEL_TYPE,value=webui_config['training']['model_type'] if webui_config['training']['model_type'] else None,interactive=True,scale=1)
                        valid_config_path = gr.Textbox(label=i18n("配置文件路径"),value=webui_config['training']['config_path'] if webui_config['training']['config_path'] else i18n("请输入配置文件路径或选择配置文件"),interactive=True,scale=3)
                    valid_model_path = gr.Textbox(label=i18n("模型路径"),value=i18n("请输入或选择模型文件"),interactive=True,scale=4)
                    valid_path = gr.Textbox(label=i18n("验证集路径"),value=webui_config['training']['valid_path'] if webui_config['training']['valid_path'] else i18n("请输入或选择验证集文件夹"),interactive=True,scale=4)
                    valid_results_path = gr.Textbox(label=i18n("输出目录"),value="results/",interactive=True,scale=3)
                    with gr.Row():
                        valid_device_ids = gr.Textbox(label=i18n("选择显卡, 多卡用户请使用空格分隔GPU ID"),value=webui_config['training']['device_ids'] if webui_config['training']['device_ids'] else "0",interactive=True)
                        valid_num_workers = gr.Number(label=i18n("验证集读取线程数, 0为自动"),value=webui_config['training']['num_workers'] if webui_config['training']['num_workers'] else 0,interactive=True,minimum=0,maximum=cpu_count(),step=1)
                        valid_extension = gr.Dropdown(label=i18n("选择验证集音频格式"),choices=["wav", "flac", "mp3"],value="wav",interactive=True,allow_custom_value=True)
                    valid_pin_memory = gr.Checkbox(label=i18n("是否将加载的数据放置在固定内存中, 默认为否"), value=webui_config['training']['pin_memory'], interactive=True)
                    valid_button = gr.Button(i18n("开始验证"), variant="primary")
                    with gr.Row():
                        valid_output_message = gr.Textbox(label="Output Message", scale=4)
                        stop_thread = gr.Button(i18n("强制停止"), scale=1)

                    valid_button.click(fn=validate_model,inputs=[valid_model_type, valid_config_path, valid_model_path, valid_path, valid_results_path, valid_device_ids, valid_num_workers, valid_extension, valid_pin_memory],outputs=valid_output_message)
                    stop_thread.click(fn=stop_all_thread)

        with gr.TabItem(label=i18n("设置")):
            with gr.Row():
                gpu_list = gr.Textbox(label=i18n("GPU信息"), value=get_device(), interactive=False)
                plantform_info = gr.Textbox(label=i18n("系统信息"), value=get_platform(), interactive=False)
                set_language = gr.Dropdown(label=i18n("选择语言"), choices=language_dict.keys(), value=get_language(), interactive=True)
            with gr.Row():
                update_message = gr.Textbox(label=i18n("检查更新"), value=i18n("当前版本: ") + PACKAGE_VERSION + i18n(", 请点击检查更新按钮"), interactive=False,scale=4)
                check_update = gr.Button(i18n("检查更新"), scale=1)
            reset_all_webui_config = gr.Button(i18n("重置WebUI路径记录"), variant="primary")
            setting_output_message = gr.Textbox(label="Output Message")

            check_update.click(fn=check_webui_update, outputs=update_message)
            reset_all_webui_config.click(fn=reset_webui_config,outputs=setting_output_message)
            set_language.change(fn=change_language,inputs=[set_language],outputs=setting_output_message)

app.launch(share=True)
