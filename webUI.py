import gradio as gr
import subprocess
import os
import sys
import re
import tempfile
import shutil
import json
import requests
import yaml
import tkinter as tk
import librosa
import numpy as np
import pandas as pd
import webbrowser
from ml_collections import ConfigDict
from tqdm import tqdm
from mir_eval.separation import bss_eval_sources
from tkinter import filedialog
from pydub import AudioSegment
# 获取当前文件的目录路径
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# 将当前工作目录更改为当前文件的目录路径
os.chdir(current_file_directory)

PACKAGE_VERSION = "1.3"
PRESETS = "data/preset_data.json"
MOSELS = "data/model_map.json"
WEBUI_CONFIG = "data/webui_config.json"
VR_MODEL = "data/vr_model.json"
AUGMENTATIONS_CONFIG = "configs_template/augmentations_template.yaml"
MODEL_FOLDER = "pretrain/"
CONFIG_TEMPLATE_FOLDER = "configs_template/"
VERSION_CONFIG = "data/version.json"
FFMPEG=".\\ffmpeg\\bin\\ffmpeg.exe"
PYTHON=".\\workenv\\python.exe"

def setup_webui():
    if os.path.exists("data"):
        if not os.path.isfile(VERSION_CONFIG):
            print("[INFO]正在初始化版本配置文件")
            shutil.copytree("configs_backup", "configs", dirs_exist_ok=True)
            shutil.copytree("configs_backup", "configs_template", dirs_exist_ok=True)
            shutil.copytree("data_backup", "data", dirs_exist_ok=True)
        else:
            version_config = load_configs(VERSION_CONFIG)
            version = version_config["version"]
            if version != PACKAGE_VERSION:
                print(f"[INFO]检测到{version}旧版配置，正在更新至最新版{PACKAGE_VERSION}")
                webui_config = load_configs(WEBUI_CONFIG)
                webui_config_backup = load_configs("data_backup/webui_config.json")
                webui_config_backup["settings"] = webui_config["settings"]
                webui_config_backup["settings_backup"] = webui_config["settings_backup"]
                save_configs(webui_config_backup, WEBUI_CONFIG)
                shutil.copytree("configs_backup", "configs", dirs_exist_ok=True)
                shutil.copytree("configs_backup", "configs_template", dirs_exist_ok=True)

                version_config["version"] = PACKAGE_VERSION
                save_configs(version_config, VERSION_CONFIG)

    if not os.path.exists("configs"):
        print("[INFO]正在初始化configs目录")
        shutil.copytree("configs_backup", "configs")
    if not os.path.exists("configs_template"):
        print("[INFO]正在初始化configs_template目录")
        shutil.copytree("configs_backup", "configs_template")
    if not os.path.exists("data"):
        print("[INFO]正在初始化data目录")
        shutil.copytree("data_backup", "data")
    if not os.path.isfile("data_backup/download_checks.json") or not os.path.isfile("data_backup/mdx_model_data.json") or not os.path.isfile("data_backup/vr_model_data.json"):
        print("[INFO]正在初始化pretrain目录")
        copy_uvr_config(os.path.join(MODEL_FOLDER, "VR_Models"))
    absolute_path = os.path.abspath("ffmpeg/bin/")
    os.environ["PATH"] += os.pathsep + absolute_path


def webui_restart():
    os.execl(PYTHON, PYTHON, *sys.argv)


def load_configs(config_path):
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

def save_configs(config, config_path):
    if config_path.endswith('.json'):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f)

def load_augmentations_config():
    try:
        with open(AUGMENTATIONS_CONFIG, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：无法找到增强配置文件模板，请检查文件{AUGMENTATIONS_CONFIG}是否存在。"


def load_msst_model():
    config = load_configs(MOSELS)
    model_list = []
    downloaded_model_list = []
    for keys in config.keys():
        for model in config[keys]:
            model_list.append(model["name"])
    model_dir = [os.path.join(MODEL_FOLDER, keys) for keys in config.keys()]
    for dirs in model_dir:
        for files in os.listdir(dirs):
            if files.endswith(('.ckpt', '.pth', '.th')) and files in model_list:
                downloaded_model_list.append(files)
    return downloaded_model_list


def get_msst_model(model_name):
    config = load_configs(MOSELS)
    for keys in config.keys():
        for model in config[keys]:
            if model["name"] == model_name:
                model_type = model["model_type"]
                model_path = os.path.join(MODEL_FOLDER, keys, model_name)
                config_path = model["config_path"]
                download_link = model["link"]
                return model_path, config_path, model_type, download_link
    raise gr.Error("模型不存在！")


def load_vr_model():
    config = load_configs(WEBUI_CONFIG)
    vr_model_path = config['settings']['uvr_model_dir']
    ckpt_files = [f for f in os.listdir(vr_model_path) if f.endswith('.pth')]
    return ckpt_files


def load_vr_model_stem(model):
    config = load_configs(VR_MODEL)
    for keys in config.keys():
        if keys == model:
            primary_stem = config[keys]["primary_stem"]
            secondary_stem = config[keys]["secondary_stem"]
            vr_primary_stem_only = gr.Checkbox(label=f"{primary_stem} Only", value=False, interactive=True)
            vr_secondary_stem_only = gr.Checkbox(label=f"{secondary_stem} Only", value=False, interactive=True)
            return vr_primary_stem_only, vr_secondary_stem_only
    raise gr.Error("模型不存在！")

def load_presets_list():
    config = load_configs(PRESETS)
    if config == {}:
        return ["无预设"]
    return list(config.keys())


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


def open_folder(folder):
    if folder == "":
        raise gr.Error("请先选择文件夹!")
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_open = folder
    absolute_path = os.path.abspath(path_to_open)
    os.system(f"explorer {absolute_path}")


def save_training_config(train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers, train_device_ids, train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path):
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
        config['training']['results_path'] = train_results_path
        save_configs(config, WEBUI_CONFIG)
        return "配置保存成功！"
    except Exception as e:
        return f"配置保存失败: {e}"


def save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
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
    config['inference']['vr_sample_rate'] = vr_sample_rate
    config['inference']['vr_batch_size'] = vr_batch_size
    config['inference']['vr_normalization'] = vr_normalization
    config['inference']['vr_post_process_threshold'] = vr_post_process_threshold
    config['inference']['vr_invert_spect'] = vr_invert_spect
    config['inference']['vr_enable_tta'] = vr_enable_tta
    config['inference']['vr_high_end_process'] = vr_high_end_process
    config['inference']['vr_enable_post_process'] = vr_enable_post_process
    config['inference']['vr_debug_mode'] = vr_debug_mode
    save_configs(config, WEBUI_CONFIG)


def save_settings(select_uvr_model_dir):
    copy_uvr_config(select_uvr_model_dir)
    config = load_configs(WEBUI_CONFIG)
    config['settings']['uvr_model_dir'] = select_uvr_model_dir
    save_configs(config, WEBUI_CONFIG)
    return "设置保存成功！请重启WebUI以应用。"


def reset_settings():
    try:
        copy_uvr_config(os.path.join(MODEL_FOLDER, "VR_Models"))
        config = load_configs(WEBUI_CONFIG)

        for key in config['settings_backup']:
            config['settings'][key] = config['settings_backup'][key]
        
        save_configs(config, WEBUI_CONFIG)
        return "设置重置成功，请重启WebUI刷新！"
    except Exception as e:
        return f"设置重置失败: {e}"

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
        return "记录重置成功，请重启WebUI刷新！"
    except Exception as e:
        return f"记录重置失败: {e}"

def copy_uvr_config(path):
    download_checks = "data_backup/download_checks.json"
    mdx_model_data = "data_backup/mdx_model_data.json"
    vr_model_data = "data_backup/vr_model_data.json"
    shutil.copy(download_checks, path)
    shutil.copy(mdx_model_data, path)
    shutil.copy(vr_model_data, path)


def init_selected_model():
    config = load_configs(WEBUI_CONFIG)
    selected_model = config['inference']['selected_model']
    if not selected_model:
        return
    _, config_path, _, _ = get_msst_model(selected_model)
    config = load_configs(config_path)
    if config.inference.get('batch_size'):
        batch_size = config.inference.get('batch_size')
    else:
        batch_size = ""
    if config.inference.get('dim_t'):
        dim_t = config.inference.get('dim_t')
    else:
        dim_t = ""
    if config.inference.get('num_overlap'):
        num_overlap = config.inference.get('num_overlap')
    else:
        num_overlap = ""
    return batch_size, dim_t, num_overlap

def init_selected_vr_model():
    webui_config = load_configs(WEBUI_CONFIG)
    config = load_configs(VR_MODEL)
    model = webui_config['inference']['vr_select_model']
    if not model:
        vr_primary_stem_only = "仅输出主音轨"
        vr_secondary_stem_only = "仅输出次音轨"
        return vr_primary_stem_only, vr_secondary_stem_only
    for keys in config.keys():
        if keys == model:
            primary_stem = config[keys]["primary_stem"]
            secondary_stem = config[keys]["secondary_stem"]
            vr_primary_stem_only = f"{primary_stem} Only"
            vr_secondary_stem_only = f"{secondary_stem} Only"
            return vr_primary_stem_only, vr_secondary_stem_only
    vr_primary_stem_only = "仅输出主音轨"
    vr_secondary_stem_only = "仅输出次音轨"
    return vr_primary_stem_only, vr_secondary_stem_only


def update_train_start_check_point(path):
    if not os.path.isdir(path):
        raise gr.Error("请先选择模型保存路径！")
    ckpt_files = [f for f in os.listdir(path) if f.endswith(('.ckpt', '.pth', '.th'))]
    return gr.Dropdown(label="初始模型", choices=ckpt_files if ckpt_files else ["None"])


def update_inference_settings(selected_model):
    _, config_path, _, _ = get_msst_model(selected_model)
    config = load_configs(config_path)
    if config.inference.get('batch_size'):
        batch_size = gr.Textbox(label="batch_size", value=str(
            config.inference.get('batch_size')), interactive=True)
    else:
        batch_size = gr.Textbox(label="batch_size", value="")
    if config.inference.get('dim_t'):
        dim_t = gr.Textbox(label="dim_t", value=str(
            config.inference.get('dim_t')), interactive=True)
    else:
            dim_t = gr.Textbox(label="dim_t", value="")
    if config.inference.get('num_overlap'):
        num_overlap = gr.Textbox(label="num_overlap", value=str(
            config.inference.get('num_overlap')), interactive=True)
    else:
        num_overlap = gr.Textbox(label="num_overlap", value="")
    return batch_size, dim_t, num_overlap


def save_config(selected_model, batch_size, dim_t, num_overlap):
    _, config_path, _, _ = get_msst_model(selected_model)
    config = load_configs(config_path)

    config.inference['batch_size'] = int(
        batch_size) if batch_size.isdigit() else None
    config.inference['dim_t'] = int(dim_t) if dim_t.isdigit() else None
    config.inference['num_overlap'] = int(
        num_overlap) if num_overlap.isdigit() else None

    save_configs(config, config_path)
    return "配置保存成功！"


def run_inference_single(selected_model, input_audio, store_dir, extract_instrumental, gpu_id, force_cpu):
    config = load_configs(WEBUI_CONFIG)
    config['inference']['selected_model'] = selected_model
    config['inference']['gpu_id'] = gpu_id
    config['inference']['force_cpu'] = force_cpu
    config['inference']['extract_instrumental'] = extract_instrumental
    config['inference']['store_dir'] = store_dir
    save_configs(config, WEBUI_CONFIG)

    if not input_audio:
        return "请上传一个音频文件。"
    input_path = os.path.dirname(input_audio)

    run_inference(selected_model, input_path, store_dir,extract_instrumental, gpu_id, force_cpu)
    return f"处理完成！分离完成的音频文件已保存在{store_dir}中。"


def run_multi_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, force_cpu):
    config = load_configs(WEBUI_CONFIG)
    config['inference']['selected_model'] = selected_model
    config['inference']['gpu_id'] = gpu_id
    config['inference']['force_cpu'] = force_cpu
    config['inference']['extract_instrumental'] = extract_instrumental
    config['inference']['store_dir'] = store_dir
    config['inference']['multiple_audio_input'] = input_folder
    save_configs(config, WEBUI_CONFIG)

    run_inference(selected_model, input_folder, store_dir,extract_instrumental, gpu_id, force_cpu)
    return f"处理完成！分离完成的音频文件已保存在{store_dir}中。"


def run_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, force_cpu):
    if not bool(re.match(r'^(\d+)(?:\s(?!\1)\d+)*$', gpu_id)):
        raise gr.Error("GPU ID格式错误，请重新输入。")
    if selected_model == "":
        raise gr.Error("请选择模型。")
    if input_folder == "":
        raise gr.Error("请选择输入目录。")
    if not os.path.exists(input_folder):
        os.makedirs(store_dir)

    start_check_point, config_path, model_type, _ = get_msst_model(selected_model)

    gpu_ids = gpu_id if not force_cpu else "0"
    extract_instrumental_option = "--extract_instrumental" if extract_instrumental else ""
    force_cpu_option = "--force_cpu" if force_cpu else ""

    command = f"{PYTHON} inference.py --model_type {model_type} --config_path \"{config_path}\" --start_check_point \"{start_check_point}\" --input_folder \"{input_folder}\" --store_dir \"{store_dir}\" --device_ids {gpu_ids} {extract_instrumental_option} {force_cpu_option}"
    print(command)
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output = ""
    while True:
        line = process.stdout.readline()
        if not line:
            break
        output += line
        print(line, end="")

    stderr = process.communicate()[1]
    if stderr:
        print(stderr)
    print(f"处理完成！分离后的音频文件已保存在{store_dir}中。")
    return output if output else stderr


def vr_inference_single(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_single_audio, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
    vr_multiple_audio_input = None
    if not os.path.isfile(vr_single_audio):
        return "请上传一个音频文件。"
    if not vr_select_model:
        return "请选择模型。"
    if not vr_store_dir:
        return "请选择输出目录。"
    if not os.path.exists(vr_store_dir):
        os.makedirs(vr_store_dir)
    save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
    message = vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_single_audio, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
    return message

def vr_inference_multi(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
    if not os.path.isdir(vr_multiple_audio_input):
        return "请选择输入文件夹。"
    if not vr_select_model:
        return "请选择模型。"
    if not vr_store_dir:
        return "请选择输出目录。"
    if not os.path.exists(vr_store_dir):
        os.makedirs(vr_store_dir)
    save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
    message = vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
    return message

def vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_audio_input, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
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
    
    sample_rate = vr_sample_rate
    use_cpu = "--use_cpu" if vr_use_cpu else ""
    vr_enable_tta = "--vr_enable_tta" if vr_enable_tta else ""
    vr_high_end_process = "--vr_high_end_process" if vr_high_end_process else ""
    vr_enable_post_process = "--vr_enable_post_process" if vr_enable_post_process else ""

    command = f"{PYTHON} uvr_inference.py \"{audio_file}\" {debug_mode} --model_filename \"{model_filename}\" --output_format {output_format} --output_dir \"{output_dir}\" --model_file_dir \"{model_file_dir}\" {invert_spect} --normalization {normalization} {single_stem} --sample_rate {sample_rate} {use_cpu} --vr_batch_size {vr_batch_size} --vr_window_size {vr_window_size} --vr_aggression {vr_aggression} {vr_enable_tta} {vr_high_end_process} {vr_enable_post_process} --vr_post_process_threshold {vr_post_process_threshold}"

    print(command)
    subprocess.run(command, shell=True)
    return f"处理完成，结果已保存至{output_dir}。"


def update_model_name(model_type):
    if model_type == "UVR_VR_Models":
        model_map = load_vr_model()
        return gr.Dropdown(label="选择模型", choices=model_map, interactive=True)
    else:
        model_map = load_msst_model()
        return gr.Dropdown(label="选择模型", choices=model_map, interactive=True)


def update_model_stem(model_type, model_name):
    if model_type == "UVR_VR_Models":
        config = load_configs(VR_MODEL)
        for keys in config.keys():
            if keys == model_name:
                primary_stem = config[keys]["primary_stem"]
                secondary_stem = config[keys]["secondary_stem"]
                return gr.Dropdown(label="stem", choices=[primary_stem, secondary_stem], interactive=True)
    else:
        return gr.Dropdown(label="stem", choices=["primary_stem"], value="primary_stem", interactive=False)


def add_to_flow_func(model_type, model_name, stem, df):
    if not model_type or not model_name:
        return df
    if model_type == "UVR_VR_Models" and not stem:
        return df
    if model_type == "UVR_VR_Models" and stem == "primary_stem":
        return df
    new_data = pd.DataFrame({"model_type": [model_type], "model_name": [model_name], "stem": [stem]})
    if df["model_type"].iloc[0] == "" or df["model_name"].iloc[0] == "":
        return new_data
    updated_df = pd.concat([df, new_data], ignore_index=True)
    return updated_df

def save_flow_func(preset_name, df):
    preset_data = load_configs(PRESETS)
    preset_dict = {row["model_name"]: row.dropna().to_dict() for _, row in df.iterrows()}
    preset_data[preset_name] = preset_dict
    save_configs(preset_data, PRESETS)

    output_message = f"预设{preset_name}保存成功"
    preset_name_delete = gr.Dropdown(label="请选择预设", choices=list(preset_data.keys()))
    preset_name_select = gr.Dropdown(label="请选择预设", choices=list(preset_data.keys()))

    return output_message, preset_name_delete, preset_name_select


def reset_flow_func():
    return gr.Dataframe(pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""]}), interactive=False, label=None)


def delete_func(preset_name):
    preset_data = load_configs(PRESETS)
    if preset_name in preset_data.keys():
        del preset_data[preset_name]
        save_configs(preset_data, PRESETS)

        output_message = f"预设{preset_name}删除成功"
        preset_name_delete = gr.Dropdown(label="请选择预设", choices=list(preset_data.keys()))
        preset_name_select = gr.Dropdown(label="请选择预设", choices=list(preset_data.keys()))
        return output_message, preset_name_delete, preset_name_select
    else:
        return "预设不存在"


def run_inference_flow(input_folder, store_dir, preset_name, force_cpu,extract_instrumental):
    preset_data = load_configs(PRESETS)
    if not preset_name in preset_data.keys():
        return f"预设'{preset_name}'不存在。"
    
    config = load_configs(WEBUI_CONFIG)
    config['inference']['preset'] = preset_name
    config['inference']['force_cpu'] = force_cpu
    config['inference']['input_folder_flow'] = input_folder
    config['inference']['store_dir_flow'] = store_dir
    save_configs(config, WEBUI_CONFIG)

    model_list = preset_data[preset_name]
    input_to_use = input_folder
    tmp_store_dir = tempfile.mkdtemp()

    for model_name in model_list.keys():
        if model_name not in load_msst_model() and model_name not in load_vr_model():
            return f"模型'{model_name}'不存在。"

    i = 0
    model_preview=""
    for model_name in model_list.keys():
        #如果前一个模型是bsr且下一个模型不是bsr时：
        if model_preview=="model_bs_roformer_ep_368_sdr_12.9628.ckpt" and model_name!="model_bs_roformer_ep_368_sdr_12.9628.ckpt":
            #获取上一个模型temp音频文件夹路径下的所有文件，例如xxx_instrumental.wav,xxx_vocals.wav
            filenames = os.listdir(tmp_store_dir)
            for filename in filenames:
                #当遍历到的文件名最后是_instrumental.wav时：
                if len(filename) >= 17 and filename[-17:] == "_instrumental.wav":
                    source_file_path = os.path.join(tmp_store_dir,filename)
                    destination_file_path = os.path.join(store_dir,filename)
                    # 剪切_instrumental.wav文件到最终文件夹
                    shutil.copy(source_file_path, destination_file_path)
                    os.remove(source_file_path)
                    shutil.rmtree(input_to_use)
                    input_to_use = tmp_store_dir
                    tmp_store_dir = tempfile.mkdtemp()
        #非前bsr，正常情况：
        else:
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
        
        print(f"Step {i+1}: Running inference using {model_name}")

        if model_list[model_name]["model_type"] == "MSST_Models":
            gpu_id = config['inference']['gpu_id'] if not force_cpu else "0"
            #extract_instrumental = True
            #这里已经将extract_instrumental作为参数传进run_inference()
            run_inference(model_name, input_to_use, tmp_store_dir, extract_instrumental, gpu_id, force_cpu)
        elif model_list[model_name]["model_type"] == "UVR_VR_Models":
            vr_model_config = load_configs(VR_MODEL)
            stem = model_list[model_name]["stem"]
            print(f"stem: {stem}")
            vr_select_model = model_name
            vr_window_size = config['inference']['vr_window_size']
            vr_aggression = config['inference']['vr_aggression']
            vr_output_format = "wav"
            vr_use_cpu = force_cpu
            vr_primary_stem_only = True if stem == vr_model_config[model_name]["primary_stem"] else False
            vr_secondary_stem_only = True if stem == vr_model_config[model_name]["secondary_stem"] else False
            vr_audio_input = input_to_use
            vr_store_dir = tmp_store_dir
            vr_sample_rate = config['inference']['vr_sample_rate']
            vr_batch_size = config['inference']['vr_batch_size']
            vr_normalization = config['inference']['vr_normalization']
            vr_post_process_threshold = config['inference']['vr_post_process_threshold']
            vr_invert_spect = config['inference']['vr_invert_spect']
            vr_enable_tta = config['inference']['vr_enable_tta']
            vr_high_end_process = config['inference']['vr_high_end_process']
            vr_enable_post_process = config['inference']['vr_enable_post_process']
            vr_debug_mode = config['inference']['vr_debug_mode']

            vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_audio_input, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
        i += 1
        model_preview=model_name

    if tmp_store_dir != store_dir:
        for file_name in os.listdir(tmp_store_dir):
            shutil.move(os.path.join(tmp_store_dir, file_name),
                        os.path.join(store_dir, file_name))
        shutil.rmtree(tmp_store_dir)

    return f"处理完成！分离完成的音频文件已保存在{store_dir}中。"


def convert_audio(uploaded_files, ffmpeg_output_format, ffmpeg_output_folder):
    if not uploaded_files:
        return "请上传至少一个文件。"
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
        try:
            subprocess.run(command, shell=True, check=True)
            success_files.append(output_file)
        except subprocess.CalledProcessError:
            print(f"转换失败: {uploaded_file_path}\n")
            continue  # 如果转换失败，则跳过当前文件

    if not success_files:
        return "所有文件转换失败，请检查文件格式和ffmpeg路径。"
    else:
        text = f"处理完成，文件已保存为：\n" + "\n".join(success_files)
        return text

def merge_audios(input_folder, output_folder):
    config = load_configs(WEBUI_CONFIG)
    config['tools']['merge_audio_input'] = input_folder
    config['tools']['merge_audio_output'] = output_folder
    save_configs(config, WEBUI_CONFIG)

    combined_audio = AudioSegment.empty()
    output_file = os.path.join(output_folder, "merged_audio.wav")
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):
            file_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_file(file_path)
            combined_audio += audio
    try:
        combined_audio.export(output_file, format="wav")
        return f"处理完成，文件已保存为：{output_file}"
    except Exception as e:
        return f"处理失败: {e}"


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
        return "请上传至少2个文件"
    if len(files) != len(weights.split()):
        return "上传的文件数目与权重数目不匹配"
    else:
        config = load_configs(WEBUI_CONFIG)
        config['tools']['ensemble_type'] = ensemble_mode
        config['tools']['ensemble_output_folder'] = output_path
        save_configs(config, WEBUI_CONFIG)

        files_argument = " ".join(files)
        output_path = os.path.join(output_path, f"ensemble_{ensemble_mode}.wav")
        command = f"{PYTHON} ensemble.py --files {files_argument} --type {ensemble_mode} --weights {weights} --output {output_path}"
        print(command)
        try:
            subprocess.run(command, shell = True)
            return f"处理完成，文件已保存为：{output_path}"
        except Exception as e:
            return f"处理失败: {e}"


def upgrade_download_model_name(model_type_dropdown):
    if model_type_dropdown == "UVR_VR_Models":
        model_map = load_configs(VR_MODEL)
        return gr.Dropdown(label="选择模型", choices=[keys for keys in model_map.keys()])
    else:
        model_map = load_configs(MOSELS)
        return gr.Dropdown(label="选择模型", choices=[model["name"] for model in model_map[model_type_dropdown]])


def download_model(model_type, model_name):
    models = load_configs(MOSELS)
    model_choices = list(models.keys())
    model_choices.append("UVR_VR_Models")
    if model_type not in model_choices:
        return "请提供模型类型和模型名称。"

    if model_type == "UVR_VR_Models":
        downloaded_model = load_vr_model()
        if model_name in downloaded_model:
            return f"模型 '{model_name}' 已安装。"

        vr_model_map = load_configs(VR_MODEL)
        if model_name not in vr_model_map.keys():
            return f"模型 '{model_name}' 不存在。"

        model_url = vr_model_map[model_name]["download_link"]
        webui_config = load_configs(WEBUI_CONFIG)
        model_path = webui_config['settings']['uvr_model_dir']
        os.makedirs(model_path, exist_ok=True)

        return download_file(model_url, os.path.join(model_path, model_name), model_name)

    presets = load_configs(MOSELS)
    model_mapping = load_msst_model()

    if model_name in model_mapping:
        return f"模型 '{model_name}' 已安装。"
    if model_type not in presets:
        return f"模型类型 '{model_type}' 不存在。"

    for model in presets[model_type]:
        if model["name"] == model_name:
            if isinstance(model["link"], str):
                model_url = model["link"]
                model_path = f"pretrain/{model_type}/{model_name}"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                download_file(model_url, model_path, model_name)
            else:
                return f"模型 '{model_name}' 的链接无效。"
    return f"模型名称 '{model_name}' 在类型 '{model_type}' 中不存在。"

def download_file(url, path, model_name):
    try:
        print(f"模型 '{model_name}' 下载中。")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(path, 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True,
                    desc=model_name, initial=0, ascii=True) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        return f"模型 '{model_name}' 下载成功。"
    except Exception as e:
        return f"模型 '{model_name}' 下载失败。错误信息: {str(e)}"

def manual_download_model(model_type, model_name):
    models = load_configs(MOSELS)
    model_choices = list(models.keys())
    model_choices.append("UVR_VR_Models")
    if model_type not in model_choices:
        return "请提供模型类型和模型名称。"

    if model_type == "UVR_VR_Models":
        downloaded_model = load_vr_model()
        if model_name in downloaded_model:
            return f"模型 '{model_name}' 已安装。"
        vr_model_map = load_configs(VR_MODEL)
        if model_name not in vr_model_map.keys():
            return f"模型 '{model_name}' 不存在。"

        model_url = vr_model_map[model_name]["download_link"]
    else:
        presets = load_configs(MOSELS)
        model_mapping = load_msst_model()

        if model_name in model_mapping:
            return f"模型 '{model_name}' 已安装。"
        if model_type not in presets:
            return f"模型类型 '{model_type}' 不存在。"

        for model in presets[model_type]:
            if model["name"] == model_name:
                if isinstance(model["link"], str):
                    model_url = model["link"]
                    break
        else:
            return f"模型名称 '{model_name}' 在类型 '{model_type}' 中不存在。"
    
    webbrowser.open(model_url)
    return f"已打开 '{model_name}' 的下载链接。"



def reset_config(selected_model):
    _, original_config_path, _, _ = get_msst_model(selected_model)
    dir_path, file_name = os.path.split(original_config_path)
    backup_dir_path = dir_path.replace("configs", "configs_backup", 1)
    backup_config_path = os.path.join(backup_dir_path, file_name)

    if os.path.exists(backup_config_path):
        shutil.copy(backup_config_path, original_config_path)
        update_inference_settings(selected_model)
        return "配置重置成功！"
    else:
        return "备份配置文件不存在。"


def start_training(train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers, train_device_ids, train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_start_check_point):
    model_type = train_model_type
    config_path = train_config_path
    start_check_point = train_start_check_point
    results_path = train_results_path
    data_path = train_dataset_path
    dataset_type = train_dataset_type
    valid_path = train_valid_path
    num_workers = train_num_workers
    device_ids = train_device_ids
    seed = train_seed
    pin_memory = train_pin_memory
    use_multistft_loss = "--use_multistft_loss" if train_use_multistft_loss else ""
    use_mse_loss = "--use_mse_loss" if train_use_mse_loss else ""
    use_l1_loss = "--use_l1_loss" if train_use_l1_loss else ""

    if model_type not in ['bs_roformer', 'mel_band_roformer', 'segm_models', 'htdemucs', 'mdx23c']:
        return "模型类型错误，请重新选择。"
    if not os.path.exists(config_path):
        return "配置文件不存在，请重新选择。"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(data_path):
        return "数据集路径不存在，请重新选择。"
    if not os.path.exists(valid_path):
        return "验证集路径不存在，请重新选择。"
    if dataset_type not in [1, 2, 3, 4]:
        return "数据集类型错误，请重新选择。"
    if num_workers < 0:
        return "num_workers不能小于0，请重新输入。"
    if not bool(re.match(r'^(\d+)(?:\s(?!\1)\d+)*$', device_ids)):
        return "device_ids格式错误，请重新输入。"
    if train_start_check_point == "None" or train_start_check_point == "":
        start_check_point = ""
    elif os.path.exists(results_path):
        start_check_point = "--start_check_point " + "\"" + os.path.join(results_path, train_start_check_point) + "\""
    else:
        return "模型保存路径不存在，请重新选择。"

    command = f"{PYTHON} train.py --model_type {model_type} --config_path \"{config_path}\" {start_check_point} --results_path \"{results_path}\" --data_path \"{data_path}\" --dataset_type {dataset_type} --valid_path \"{valid_path}\" --num_workers {num_workers} --device_ids {device_ids} --seed {seed} --pin_memory {pin_memory} {use_multistft_loss} {use_mse_loss} {use_l1_loss}"
    print(command)
    try:
        subprocess.run(command, shell=True)
        # 按道理这边会阻塞住，如果下面的return被执行，说明大概率是出错了（也有可能训练结束）
        return "请前往控制台查看报错信息！"
    except Exception as e:
        return f"训练启动失败: {e}"


with gr.Blocks(
        theme=gr.Theme.load('themes/theme_schema@1.2.2.json')
) as app:

    gr.Markdown(value=f"""
    ### Music-Source-Separation-Training-Inference-Webui v{PACKAGE_VERSION}

    仅供个人娱乐和非商业用途，禁止用于血腥、暴力、性相关、政治相关内容。<br>
    本整合包完全免费，严禁以任何形式倒卖，如果你从任何地方**付费**购买了本整合包，请**立即退款**。<br> 
    整合包作者：[bilibili@阿狸不吃隼舞](https://space.bilibili.com/403335715) [github@KitsuneX07](https://github.com/KitsuneX07) | [Bilibili@Sucial丶](https://space.bilibili.com/445022409) [Github@SUC-DriverOld](https://github.com/SUC-DriverOld) Gradio主题来自 [https://huggingface.co/spaces/NoCrypt/miku](https://huggingface.co/spaces/NoCrypt/miku)
    """)
    with gr.Tabs():
        setup_webui()

        webui_config = load_configs(WEBUI_CONFIG)
        presets = load_configs(PRESETS)
        models = load_configs(MOSELS)
        vr_model = load_configs(VR_MODEL)

        with gr.TabItem(label="MSST分离"):
            gr.Markdown(value="""MSST音频分离原项目地址：[https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)""")

            selected_model = gr.Dropdown(
                label="选择模型",
                choices=load_msst_model(),
                value=webui_config['inference']['selected_model'] if webui_config['inference']['selected_model'] else None,
                interactive=True
            )

            gr.Markdown(value="""多卡用户请使用空格分隔GPU ID，例如：``0 1 2``。""")
            with gr.Row():
                gpu_id = gr.Textbox(
                    label="选择使用的GPU ID",
                    value=webui_config['inference']['gpu_id'] if webui_config['inference']['gpu_id'] else "0",
                    interactive=True
                )
                with gr.Column():
                    force_cpu = gr.Checkbox(
                        label="使用CPU（注意: 使用CPU会导致速度非常慢）",
                        value=webui_config['inference']['force_cpu'] if webui_config['inference']['force_cpu'] else False,
                        interactive=True
                    )
                    extract_instrumental = gr.Checkbox(
                        label="保留次级输出（对于Vocals-Instruments类模型，勾选此项会同时输出伴奏）",
                        value=webui_config['inference']['extract_instrumental'] if webui_config['inference']['extract_instrumental'] else False,
                        interactive=True
                    )

            with gr.Tabs():
                with gr.TabItem(label="单个音频上传"):
                    single_audio = gr.Audio(label="单个音频上传", type="filepath")
                with gr.TabItem(label="批量音频上传"):
                    with gr.Row():
                        multiple_audio_input = gr.Textbox(
                            label="输入的音频目录",
                            value=webui_config['inference']['multiple_audio_input'] if webui_config[
                                'inference']['multiple_audio_input'] else "input/",
                            interactive=True,
                            scale=3
                        )
                        select_multi_input_dir = gr.Button("选择文件夹", scale=1)
                        open_multi_input_dir = gr.Button("打开文件夹", scale=1)
            with gr.Row():
                store_dir = gr.Textbox(
                    label="输出目录",
                    value=webui_config['inference']['store_dir'] if webui_config['inference']['store_dir'] else "results/",
                    interactive=True,
                    scale=3)
                select_store_btn = gr.Button("选择文件夹", scale=1)
                open_store_btn = gr.Button("打开文件夹", scale=1)

            with gr.Row():
                inference_single = gr.Button("单个音频分离", variant="primary")
                inference_multiple = gr.Button("批量音频分离", variant="primary")

            with gr.Accordion("推理参数设置（一般不需要动）", open=False):
                gr.Markdown(value="""
                只有在点击保存后才会生效。参数直接写入配置文件，无法撤销。假如不知道如何设置，请保持默认值。<br>
                请牢记自己修改前的参数数值，防止出现问题以后无法恢复。<br>
                请确保输入正确的参数，否则可能会导致模型无法正常运行。<br>
                假如修改后无法恢复，请点击``重置``按钮，这会使得配置文件恢复到默认值。<br>

                ### 参数说明

                * batch_size：批次大小，一般不需要改
                * dim_t：时序维度大小，一般不需要改
                * num_overlap：窗口重叠长度，也可理解为每帧推理的次数，数值越小速度越快，但会牺牲效果
                """)
                if webui_config['inference']['selected_model']:
                    batch_size_number, dim_t_number, num_overlap_number = init_selected_model()
                else:
                    batch_size_number, dim_t_number, num_overlap_number = "请先选择模型", "请先选择模型", "请先选择模型"
                batch_size = gr.Textbox(label="batch_size", value=batch_size_number)
                dim_t = gr.Textbox(label="dim_t", value=dim_t_number)
                num_overlap = gr.Textbox(label="num_overlap", value=num_overlap_number)
                reset_config_button = gr.Button("重置配置", variant="secondary")
                save_config_button = gr.Button("保存配置", variant="primary")

            output_message = gr.Textbox(label="Output Message")

            inference_single.click(
                fn=run_inference_single,
                inputs=[selected_model, single_audio, store_dir, extract_instrumental, gpu_id, force_cpu],
                outputs=output_message
            )
            inference_multiple.click(
                fn=run_multi_inference, inputs=[selected_model, multiple_audio_input, store_dir, extract_instrumental, gpu_id, force_cpu],
                outputs=output_message
            )
            selected_model.change(
                fn=update_inference_settings,
                inputs=[selected_model],
                outputs=[batch_size, dim_t, num_overlap]
            )
            save_config_button.click(
                fn=save_config,
                inputs=[selected_model, batch_size, dim_t, num_overlap],
                outputs=output_message
            )
            reset_config_button.click(
                fn=reset_config,
                inputs=[selected_model],
                outputs=output_message
            )
            select_store_btn.click(fn=select_folder, outputs=store_dir)
            open_store_btn.click(fn=open_folder, inputs=store_dir)
            select_multi_input_dir.click(fn=select_folder, outputs=multiple_audio_input)
            open_multi_input_dir.click(fn=open_folder, inputs=multiple_audio_input)

        with gr.TabItem(label="UVR分离"):
            gr.Markdown(value="""说明：本整合包仅融合了UVR的VR Architecture模型，MDX23C和HtDemucs类模型可以直接使用前面的MSST音频分离。<br>
            使用UVR模型进行音频分离时，若有可用的GPU，软件将自动选择，否则将使用CPU进行分离。<br>
            UVR分离使用项目：[https://github.com/nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 并进行了优化。
            """)
            vr_select_model = gr.Dropdown(
                label="选择模型", 
                choices=load_vr_model(), 
                value=webui_config['inference']['vr_select_model'] if webui_config['inference']['vr_select_model'] else None,
                interactive=True
                )
            with gr.Row():
                vr_window_size = gr.Dropdown(
                    label="Window Size：窗口大小，用于平衡速度和质量", 
                    choices=["320", "512", "1024"], 
                    value=webui_config['inference']['vr_window_size'] if webui_config['inference']['vr_window_size'] else "512", 
                    interactive=True
                    )
                vr_aggression = gr.Number(
                    label="Aggression：主干提取强度，范围-100-100，人声请选5", 
                    minimum=-100, 
                    maximum=100, 
                    value=webui_config['inference']['vr_aggression'] if webui_config['inference']['vr_aggression'] else 5, 
                    interactive=True
                    )
                vr_output_format = gr.Dropdown(
                    label="输出格式", 
                    choices=["wav", "flac", "mp3"], 
                    value=webui_config['inference']['vr_output_format'] if webui_config['inference']['vr_output_format'] else "wav", 
                    interactive=True
                    )
            with gr.Row():
                vr_primary_stem_label, vr_secondary_stem_label = init_selected_vr_model()
                vr_use_cpu = gr.Checkbox(
                    label="使用CPU", 
                    value=webui_config['inference']['vr_use_cpu'] if webui_config['inference']['vr_use_cpu'] else False, 
                    interactive=True
                    )
                vr_primary_stem_only = gr.Checkbox(
                    label=vr_primary_stem_label, 
                    value=webui_config['inference']['vr_primary_stem_only'] if webui_config['inference']['vr_primary_stem_only'] else False, 
                    interactive=True
                    )
                vr_secondary_stem_only = gr.Checkbox(
                    label=vr_secondary_stem_label, 
                    value=webui_config['inference']['vr_secondary_stem_only'] if webui_config['inference']['vr_secondary_stem_only'] else False, 
                    interactive=True
                    )
            with gr.Tabs():
                with gr.TabItem(label="单个音频上传"):
                    vr_single_audio = gr.Audio(label="单个音频上传", type="filepath")
                with gr.TabItem(label="批量音频上传"):
                    with gr.Row():
                        vr_multiple_audio_input = gr.Textbox(
                            label="输入的音频目录",
                            value=webui_config['inference']['vr_multiple_audio_input'] if webui_config['inference']['vr_multiple_audio_input'] else "input/",
                            interactive=True,
                            scale=3
                        )
                        vr_select_multi_input_dir = gr.Button("选择文件夹", scale=1)
                        vr_open_multi_input_dir = gr.Button("打开文件夹", scale=1)
            with gr.Row():
                vr_store_dir = gr.Textbox(
                    label="输出目录",
                    value=webui_config['inference']['vr_store_dir'] if webui_config['inference']['vr_store_dir'] else "results/",
                    interactive=True,
                    scale=3
                    )
                vr_select_store_btn = gr.Button("选择文件夹", scale=1)
                vr_open_store_btn = gr.Button("打开文件夹", scale=1)
            with gr.Accordion("以下是一些高级设置，一般保持默认即可", open=False):
                with gr.Row():
                    vr_sample_rate = gr.Dropdown(
                        label="Sample Rate：输出音频的采样率，可选的值有32000、44100、48000", 
                        choices=["32000", "44100", "48000"], 
                        value=webui_config['inference']['vr_sample_rate'] if webui_config['inference']['vr_sample_rate'] else "44100", 
                        interactive=True
                        )
                    vr_batch_size = gr.Number(
                        label="Batch Size：一次要处理的批次数，越大占用越多RAM，处理速度加快", 
                        minimum=1, 
                        value=webui_config['inference']['vr_batch_size'] if webui_config['inference']['vr_batch_size'] else 4, 
                        interactive=True
                        )
                    vr_normalization = gr.Number(
                        label="Normalization：最大峰值振幅，用于归一化输入和输出音频。取值为0-1", 
                        minimum=0.0, 
                        maximum=1.0, 
                        step=0.01, 
                        value=webui_config['inference']['vr_normalization'] if webui_config['inference']['vr_normalization'] else 0.9, 
                        interactive=True
                        )
                    vr_post_process_threshold = gr.Number(
                        label="Post Process Threshold：后处理特征阈值，取值为0.1-0.3", 
                        minimum=0.1, 
                        maximum=0.3, 
                        step=0.01, 
                        value=webui_config['inference']['vr_post_process_threshold'] if webui_config['inference']['vr_post_process_threshold'] else 0.2, 
                        interactive=True
                        )
                with gr.Row():
                    vr_invert_spect = gr.Checkbox(
                        label="Invert Spectrogram：二级步骤将使用频谱图而非波形进行反转，可能会提高质量，但速度稍慢", 
                        value=webui_config['inference']['vr_invert_spect'] if webui_config['inference']['vr_invert_spect'] else False, 
                        interactive=True
                        )
                    vr_enable_tta = gr.Checkbox(
                        label="Enable TTA：启用“测试时间增强”，可能会提高质量，但速度稍慢", 
                        value=webui_config['inference']['vr_enable_tta'] if webui_config['inference']['vr_enable_tta'] else False, 
                        interactive=True
                        )
                    vr_high_end_process = gr.Checkbox(
                        label="High End Process：将输出音频缺失的频率范围镜像输出", 
                        value=webui_config['inference']['vr_high_end_process'] if webui_config['inference']['vr_high_end_process'] else False, 
                        interactive=True
                        )
                    vr_enable_post_process = gr.Checkbox(
                        label="Enable Post Process：识别人声输出中残留的人工痕迹，可改善某些歌曲的分离效果", 
                        value=webui_config['inference']['vr_enable_post_process'] if webui_config['inference']['vr_enable_post_process'] else False, 
                        interactive=True
                        )
                vr_debug_mode = gr.Checkbox(
                    label="Debug Mode：启用调试模式，向开发人员反馈时，请开启此模式", 
                    value=webui_config['inference']['vr_debug_mode'] if webui_config['inference']['vr_debug_mode'] else False, 
                    interactive=True
                    )
            with gr.Row():
                vr_start_single_inference = gr.Button("开始单个分离", variant="primary")
                vr_start_multi_inference = gr.Button("开始批量分离", variant="primary")
            vr_output_message = gr.Textbox(label="Output Message")

            vr_select_model.change(
                fn=load_vr_model_stem,
                inputs=vr_select_model,
                outputs=[vr_primary_stem_only, vr_secondary_stem_only])
            vr_select_multi_input_dir.click(fn=select_folder, outputs=vr_multiple_audio_input)
            vr_open_multi_input_dir.click(fn=open_folder, inputs=vr_multiple_audio_input)
            vr_select_store_btn.click(fn=select_folder, outputs=vr_store_dir)
            vr_open_store_btn.click(fn=open_folder, inputs=vr_store_dir)
            vr_start_single_inference.click(
                fn=vr_inference_single,
                inputs=[vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_single_audio, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode],
                outputs=vr_output_message
            )
            vr_start_multi_inference.click(
                fn=vr_inference_multi,
                inputs=[vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_sample_rate, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode],
                outputs=vr_output_message
            )

        with gr.TabItem(label="预设流程"):
            gr.Markdown(value="""
                预设流程允许按照预设的顺序运行多个模型。每一个模型的输出将作为下一个模型的输入。<br>
                """)
            with gr.Tabs():
                with gr.TabItem(label="使用预设"):
                    preset_dropdown = gr.Dropdown(
                        label="请选择预设",
                        choices=list(presets.keys()),
                        value=webui_config['inference']['preset'] if webui_config['inference']['preset'] else None,
                        interactive=True
                    )

                    force_cpu = gr.Checkbox(
                        label="使用CPU（注意: 使用CPU会导致速度非常慢）",
                        value=webui_config['inference']['force_cpu'] if webui_config['inference']['force_cpu'] else False,
                        interactive=True
                    )
                    extract_instrumental = gr.Checkbox(
                        label="保留次级输出（对于Vocals-Instruments类模型，勾选此项会同时输出伴奏）",
                        value=webui_config['inference']['extract_instrumental'] if webui_config['inference']['extract_instrumental'] else False,
                        interactive=True
                    )

                    with gr.Row():
                        input_folder_flow = gr.Textbox(
                            label="输入目录",
                            value=webui_config['inference']['input_folder_flow'] if webui_config['inference']['input_folder_flow'] else "input/",
                            interactive=True,
                            scale=3
                        )
                        select_input_dir = gr.Button("选择文件夹", scale=1)
                        open_input_dir = gr.Button("打开文件夹", scale=1)
                    with gr.Row():
                        store_dir_flow = gr.Textbox(
                            label="输出目录",
                            value=webui_config['inference']['store_dir_flow'] if webui_config['inference']['store_dir_flow'] else "results/",
                            interactive=True,
                            scale=3
                        )
                        select_output_dir = gr.Button("选择文件夹", scale=1)
                        open_output_dir = gr.Button("打开文件夹", scale=1)
                    inference_flow = gr.Button("运行预设流程", variant="primary")
                    gr.Markdown(value="""
                    该模式下的UVR推理参数将直接沿用UVR分离页面的推理参数，如需修改请前往UVR分离页面。
                    """)
                    output_message_flow = gr.Textbox(label="Output Message")
                
                with gr.TabItem(label="制作预设"):
                    gr.Markdown("""
                        注意：MSST模型仅支持输出主要音轨，UVR模型支持自定义输出。<br>
                        """)
                    preset_name_input = gr.Textbox(label="预设名称", placeholder="请输入预设名称", interactive=True)
                    with gr.Row():
                        model_type = gr.Dropdown(label="model_type", choices=["MSST_Models", "UVR_VR_Models"], interactive=True)
                        model_name = gr.Dropdown(label="model_name", choices=["请先选择模型类型"], interactive=True)
                        stem = gr.Dropdown(label="stem", choices=["请先选择模型"], interactive=True)
                    add_to_flow = gr.Button("添加至流程")
                    gr.Markdown("""预设流程""")
                    preset_flow = gr.Dataframe(pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""]}), interactive=False, label=None)
                    reset_flow = gr.Button("重新输入")
                    save_flow = gr.Button("保存上述预设流程", variant="primary")
                    gr.Markdown("""删除预设""")
                    with gr.Row():
                        preset_name_delete = gr.Dropdown(label="请选择预设", choices=load_presets_list(), interactive=True)
                    delete_button = gr.Button("删除所选预设", scale=1)
                    output_message_make = gr.Textbox(label="Output Message")

            inference_flow.click(
                fn=run_inference_flow,
                inputs=[input_folder_flow, store_dir_flow,
                        preset_dropdown, force_cpu,extract_instrumental],
                outputs=output_message_flow
            )

            select_input_dir.click(fn=select_folder, outputs=input_folder_flow)
            open_input_dir.click(fn=open_folder, inputs=input_folder_flow)
            select_output_dir.click(fn=select_folder, outputs=store_dir_flow)
            open_output_dir.click(fn=open_folder, inputs=store_dir_flow)

            model_type.change(update_model_name, inputs=model_type, outputs=model_name)
            model_name.change(update_model_stem, inputs=[model_type, model_name], outputs=stem)
            add_to_flow.click(add_to_flow_func, [model_type, model_name, stem, preset_flow], preset_flow)
            save_flow.click(save_flow_func, [preset_name_input, preset_flow], [output_message_make, preset_name_delete, preset_dropdown])
            reset_flow.click(reset_flow_func, [], preset_flow)
            delete_button.click(delete_func, [preset_name_delete], [output_message_make, preset_name_delete, preset_dropdown])

        with gr.TabItem(label="小工具"):
            with gr.Tabs():
                with gr.TabItem(label="音频格式转换"):
                    gr.Markdown(value="""
                        上传一个或多个音频文件并将其转换为指定格式。<br>
                        支持的格式包括 .mp3, .flac, .wav, .ogg, .m4a, .wma, .aac...等等。<br>
                        **不支持**网易云音乐、QQ音乐等加密格式，如.ncm, .qmc等。<br>
                        """)
                    with gr.Row():
                        inputs = gr.Files(label="上传音频文件")
                        with gr.Column():
                            ffmpeg_output_format = gr.Dropdown(
                                label="选择或输入音频输出格式",
                                choices=["wav", "flac", "mp3", "ogg", "m4a", "wma", "aac"],
                                value=webui_config['tools']['ffmpeg_output_format'] if webui_config['tools']['ffmpeg_output_format'] else "wav",
                                allow_custom_value=True,
                                interactive=True
                                )
                            ffmpeg_output_folder = gr.Textbox(
                                label="选择音频输出目录", 
                                value=webui_config['tools']['ffmpeg_output_folder'] if webui_config['tools']['ffmpeg_output_folder'] else "results/ffmpeg_output/", 
                                interactive=True
                                )
                            select_ffmpeg_output_dir = gr.Button("选择文件夹")
                            open_ffmpeg_output_dir = gr.Button("打开文件夹")
                    convert_audio_button = gr.Button("转换音频", variant="primary")
                    output_message_ffmpeg = gr.Textbox(label="Output Message")

                    convert_audio_button.click(
                        fn=convert_audio, 
                        inputs=[inputs, ffmpeg_output_format, ffmpeg_output_folder], 
                        outputs=output_message_ffmpeg
                        )
                    select_ffmpeg_output_dir.click(fn=select_folder, outputs=ffmpeg_output_folder)
                    open_ffmpeg_output_dir.click(fn=open_folder, inputs=ffmpeg_output_folder)
                
                with gr.TabItem(label="合并音频"):
                    gr.Markdown(value="""
                        点击合并音频按钮后，将自动把输入文件夹中的所有音频文件合并为一整个音频文件<br>
                        目前支持的格式包括 .mp3, .flac, .wav, .ogg 这四种<br>
                        合并后的音频会保存至输出目录中，文件名为merged_audio.wav<br>
                        """)
                    with gr.Row():
                        merge_audio_input = gr.Textbox(
                            label="输入的音频目录",
                            value=webui_config['tools']['merge_audio_input'] if webui_config['tools']['merge_audio_input'] else "input/",
                            interactive=True,
                            scale=3
                        )
                        select_merge_input_dir = gr.Button("选择文件夹", scale=1)
                        open_merge_input_dir = gr.Button("打开文件夹", scale=1)
                    with gr.Row():
                        merge_audio_output = gr.Textbox(
                            label="输出目录",
                            value=webui_config['tools']['merge_audio_output'] if webui_config['tools']['merge_audio_output'] else "results/merge",
                            interactive=True,
                            scale=3
                        )
                        select_merge_output_dir = gr.Button("选择文件夹", scale=1)
                        open_merge_output_dir = gr.Button("打开文件夹", scale=1)
                    merge_audio_button = gr.Button("合并音频", variant="primary")
                    output_message_merge = gr.Textbox(label="Output Message")

                    merge_audio_button.click(merge_audios, [merge_audio_input, merge_audio_output], outputs=output_message_merge)
                    select_merge_input_dir.click(fn=select_folder, outputs=merge_audio_input)
                    open_merge_input_dir.click(fn=open_folder, inputs=merge_audio_input)
                    select_merge_output_dir.click(fn=select_folder, outputs=merge_audio_output)
                    open_merge_output_dir.click(fn=open_folder, inputs=merge_audio_output)

                with gr.TabItem(label="计算SDR"):
                    with gr.Column():
                        gr.Markdown(value="""
                        上传两个音频文件并计算它们的[SDR](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021#evaluation-metric)。<br>
                        """)
                    with gr.Row():
                        true_audio = gr.Audio(label="原始音频", type="filepath")
                        estimated_audio = gr.Audio(
                            label="分离后的音频", type="filepath")
                        output_message_sdr = gr.Textbox(label="Output Message")

                    compute_sdr_button = gr.Button("计算SDR", variant="primary")

                    compute_sdr_button.click(
                        process_audio, [true_audio, estimated_audio], outputs=output_message_sdr)

                with gr.TabItem(label = "Ensemble模式"):
                    gr.Markdown(value = """
                        可用于集成不同算法的结果。<br>
                        具体的文档位于/docs/ensemble.md。<br>
                        ### 集成类型：
                        * `avg_wave` - 在 1D 变体上进行集成，独立地找到波形的每个样本的平均值
                        * `median_wave` - 在 1D 变体上进行集成，独立地找到波形的每个样本的中位数
                        * `min_wave` - 在 1D 变体上进行集成，独立地找到波形的每个样本的最小绝对值
                        * `max_wave` - 在 1D 变体上进行集成，独立地找到波形的每个样本的最大绝对值
                        * `avg_fft` - 在频谱图（短时傅里叶变换（STFT），2D 变体）上进行集成，独立地找到频谱图的每个像素的平均值。平均后使用逆 STFT 得到原始的 1D 波形。
                        * `median_fft` - 与 avg_fft 相同，但使用中位数代替平均值（仅在集成 3 个或更多来源时有用）。
                        * `min_fft` - 与 avg_fft 相同，但使用最小函数代替平均值（减少激进程度）。
                        * `max_fft` - 与 avg_fft 相同，但使用最大函数代替平均值（增加激进程度）。
                        ### 注：
                        * min_fft 可用于进行更保守的合成 - 它将减少更激进模型的影响。
                        * 最好合成等质量的模型 - 在这种情况下，它将带来增益。如果其中一个模型质量不好 - 它将降低整体质量。
                        * 在原仓库作者的实验中，与其他方法相比，avg_wave 在 SDR 分数上总是更好或相等。
                        * 上传的文件名**不能包含空格**，最终会在输出目录下生成一个ensemble_<集成模式>.wav。
                        """)
                    with gr.Row():
                        files = gr.Files(label = "上传多个音频文件", type = "filepath", file_count = 'multiple')
                        with gr.Column():
                            with gr.Row():
                                ensemble_type = gr.Dropdown(
                                    choices = ["avg_wave", "median_wave", "min_wave", "max_wave", "avg_fft", "median_fft", "min_fft", "max_fft"],
                                    label = "集成模式",
                                    value = webui_config['tools']['ensemble_type'] if webui_config['tools']['ensemble_type'] else "avg_wave",
                                    interactive=True
                                    )
                                weights = gr.Textbox(label = "权重(以空格分隔，数量要与上传的音频一致)", value = "1 1")
                            ensembl_output_path = gr.Textbox(
                            label = "输出目录", 
                            value = webui_config['tools']['ensemble_output_folder'] if webui_config['tools']['ensemble_output_folder'] else "results/ensemble/",
                            interactive=True
                            )
                            select_ensembl_output_path = gr.Button("选择文件夹")
                            open_ensembl_output_path = gr.Button("打开文件夹")
                    ensemble_button = gr.Button("运行", variant = "primary")
                    output_message_ensemble = gr.Textbox(label = "Output Message")

                    ensemble_button.click(
                        fn = ensemble, 
                        inputs = [files, ensemble_type, weights, ensembl_output_path],
                        outputs = output_message_ensemble
                        )
                    select_ensembl_output_path.click(fn = select_folder, outputs = ensembl_output_path)
                    open_ensembl_output_path.click(fn = open_folder, inputs = ensembl_output_path)

        with gr.TabItem(label="安装模型"):
            uvr_model_folder = webui_config['settings']['uvr_model_dir']
            gr.Markdown(value=f"""
            自动从huggingface镜像站或Github下载模型，无需手动下载。<br>
            若自动下载出现报错或下载过慢，请点击手动下载，跳转至下载链接，下载完成后按照指示放置在指定目录。
            ### 注意：
            * MSST模型默认下载在pretrain/<模型类型>文件夹下。
            * UVR模型默认下载在设置中的UVR模型目录中。
            * **请勿删除**UVR模型目录下的download_checks.json，mdx_model_data.json，vr_model_data.json这三个文件！
            * 需要重启WebUI才能刷新新模型哦。
            ### 当前UVR模型目录：{uvr_model_folder}，如需更改，请前往设置页面。
            ### 手动下载完成后，请根据你上面选择的模型类型放置到对应文件夹内。<br>
            """)
            with gr.Row():
                model_choices = list(models.keys())
                model_choices.append("UVR_VR_Models")
                model_type_dropdown = gr.Dropdown(
                    label="模型类型", choices=model_choices)
                download_model_name_dropdown = gr.Dropdown(
                    label="选择模型", choices=["请先选择模型类型。"])
            with gr.Row():
                open_model_dir = gr.Button("打开MSST模型目录")
                open_uvr_model_dir = gr.Button("打开UVR模型目录")
            with gr.Row():
                download_button = gr.Button("自动下载", variant="primary")
                manual_download_button = gr.Button("手动下载", variant="primary")
            output_message_download = gr.Textbox(label="Output Message")
            gr.Markdown(value="""
                下加载进度可以打开终端查看。如果一直卡着不动或者速度很慢，在确信网络正常的情况下请尝试重启webui。<br>
                若下载失败，**会在模型目录留下一个损坏的模型，请务必打开模型目录手动删除！**<br>
                下面是一些模型下载链接：<br>
                huggingface镜像站: <https://hf-mirror.com/KitsuneX07/Music_Source_Sepetration_Models><br>
                huggingface: <https://huggingface.co/KitsuneX07/Music_Source_Sepetration_Models><br>
                UVR模型仓库地址：<https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models><br>
                """)
            restart_webui = gr.Button("重启WebUI", variant="primary")
            gr.Markdown('''点击“重启WebUI”按钮后，会短暂性的失去连接，随后会自动开启一个新网页。''')

            model_type_dropdown.change(
                fn=upgrade_download_model_name,
                inputs=[model_type_dropdown],
                outputs=[download_model_name_dropdown]
            )
            download_button.click(
                fn=download_model,
                inputs=[model_type_dropdown, download_model_name_dropdown],
                outputs=output_message_download
            )
            manual_download_button.click(
                fn=manual_download_model,
                inputs=[model_type_dropdown, download_model_name_dropdown],
                outputs=output_message_download
            )
            open_model_dir.click(open_folder, inputs=gr.Textbox(MODEL_FOLDER, visible=False))
            open_uvr_model_dir.click(open_folder, inputs=gr.Textbox(uvr_model_folder, visible=False))
            restart_webui.click(webui_restart)

        with gr.TabItem(label="MSST训练"):
            gr.Markdown(value="""
                此页面提供数据集制作教程，训练参数选择，以及一键训练。有关配置文件的修改和数据集文件夹的详细说明请参考MSST原项目:[https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)<br>
                在开始下方的模型训练之前，请先进行训练数据的制作。<br>
                """)
            with gr.Tabs():
                with gr.TabItem(label="训练模型"):
                    with gr.Row():
                        train_model_type = gr.Dropdown(
                            label="选择训练模型类型",
                            choices=['bs_roformer', 'mel_band_roformer', 'segm_models', 'htdemucs', 'mdx23c'],
                            value=webui_config['training']['model_type'] if webui_config['training']['model_type'] else None,
                            interactive=True,
                            scale=1
                        )
                        train_config_path = gr.Textbox(
                            label="配置文件路径",
                            value=webui_config['training']['config_path'] if webui_config[
                                'training']['config_path'] else "请输入配置文件路径或选择配置文件",
                            interactive=True,
                            scale=3
                        )
                        select_train_config_path = gr.Button("选择配置文件", scale=1)
                    with gr.Row():
                        train_dataset_type = gr.Dropdown(
                            label="数据集类型",
                            choices=[1, 2, 3, 4],
                            value=webui_config['training']['dataset_type'] if webui_config['training']['dataset_type'] else None,
                            interactive=True,
                            scale=1
                        )
                        train_dataset_path = gr.Textbox(
                            label="数据集路径",
                            value=webui_config['training']['dataset_path'] if webui_config[
                                'training']['dataset_path'] else "请输入或选择数据集文件夹",
                            interactive=True,
                            scale=3
                        )
                        select_train_dataset_path = gr.Button(
                            "选择数据集文件夹", scale=1)
                    gr.Markdown(value="""
                        说明：数据集类型即训练集制作Step 1中你选择的类型，1：Type1；2：Type2；3：Type3；4：Type4，必须与你的数据集类型相匹配。
                        """)
                    with gr.Row():
                        train_valid_path = gr.Textbox(
                            label="验证集路径",
                            value=webui_config['training']['valid_path'] if webui_config['training']['valid_path'] else "请输入或选择验证集文件夹",
                            interactive=True,
                            scale=4)
                        select_train_valid_path = gr.Button(
                            "选择验证集文件夹", scale=1)
                    gr.Markdown(value="""
                        以下是一些训练参数的设置
                        device_ids: 选择显卡，多卡用户请使用空格分隔GPU ID，例如：`0 1 2`。
                        """)
                    with gr.Row():
                        train_num_workers = gr.Number(
                            label="num_workers: 数据集读取线程数，默认为0",
                            value=webui_config['training']['num_workers'] if webui_config['training']['num_workers'] else 0,
                            interactive=True,
                        )
                        train_device_ids = gr.Textbox(
                            label="device_ids：选择显卡",
                            value=webui_config['training']['device_ids'] if webui_config['training']['device_ids'] else "0",
                            interactive=True)
                        train_seed = gr.Number(label="随机数种子，0为随机", value="0")
                    with gr.Row():
                        train_pin_memory = gr.Checkbox(
                            label="是否将加载的数据放置在固定内存中，默认为否", value=webui_config['training']['pin_memory'], interactive=True)
                        train_use_multistft_loss = gr.Checkbox(
                            label="是否使用MultiSTFT Loss，默认为否", value=webui_config['training']['use_multistft_loss'], interactive=True)
                        train_use_mse_loss = gr.Checkbox(
                            label="是否使用MSE loss，默认为否", value=webui_config['training']['use_mse_loss'], interactive=True)
                        train_use_l1_loss = gr.Checkbox(
                            label="是否使用L1 loss，默认为否", value=webui_config['training']['use_l1_loss'], interactive=True)
                    with gr.Row():
                        train_results_path = gr.Textbox(
                            label="模型保存路径",
                            value=webui_config['training']['results_path'] if webui_config[
                                'training']['results_path'] else "请输入或选择模型保存文件夹",
                            interactive=True,
                            scale=3)
                        select_train_results_path = gr.Button("选择文件夹", scale=1)
                        open_train_results_path = gr.Button("打开文件夹", scale=1)
                    gr.Markdown(value="""
                        说明：继续训练或微调模型训练时，请选择初始模型，否则将从头开始训练！
                        """)
                    with gr.Row():
                        train_start_check_point = gr.Dropdown(
                            label="初始模型", choices=["None"], value="None", interactive=True, scale=4)
                        reflesh_start_check_point = gr.Button(
                            "刷新初始模型列表", scale=1)
                    save_train_config = gr.Button("保存上述训练配置")
                    start_train_button = gr.Button("开始训练", variant="primary")
                    gr.Markdown(value="""
                        点击开始训练后，请到终端查看训练进度或报错，下方不会输出报错信息，想要停止训练可以直接关闭终端。在训练过程中，你也可以关闭网页，仅**保留终端**。<br>
                        """)
                    output_message_train = gr.Textbox(label="Output Message")

                    select_train_config_path.click(
                        fn=select_yaml_file, outputs=train_config_path)
                    select_train_dataset_path.click(
                        fn=select_folder, outputs=train_dataset_path)
                    select_train_valid_path.click(
                        fn=select_folder, outputs=train_valid_path)
                    select_train_results_path.click(
                        fn=select_folder, outputs=train_results_path)
                    open_train_results_path.click(
                        fn=open_folder, inputs=train_results_path)
                    save_train_config.click(
                        fn=save_training_config,
                        inputs=[train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers,
                                train_device_ids, train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path],
                        outputs=output_message_train)
                    start_train_button.click(
                        fn=start_training,
                        inputs=[train_model_type, train_config_path, train_dataset_type, train_dataset_path, train_valid_path, train_num_workers, train_device_ids,
                                train_seed, train_pin_memory, train_use_multistft_loss, train_use_mse_loss, train_use_l1_loss, train_results_path, train_start_check_point],
                        outputs=output_message_train
                    )
                    reflesh_start_check_point.click(
                        fn=update_train_start_check_point,
                        inputs=train_results_path,
                        outputs=train_start_check_point
                    )

                with gr.TabItem(label="训练集制作"):
                    with gr.Accordion("Step 1: 数据集制作", open=False):
                        gr.Markdown(value="""
                            请**任选下面四种类型之一**制作数据集文件夹，并按照给出的目录层级放置你的训练数据。完成后，记录你的数据集**文件夹路径**以及你选择的**数据集类型**，以便后续使用。
                            """)
                        with gr.Row():
                            gr.Markdown("""
                                # Type 1 (MUSDB)
                                
                                不同的文件夹。每个文件夹包含所需的所有stems，格式为stem_name.wav。与MUSDBHQ18数据集相同。在最新的代码版本中，可以使用flac替代wav。<br>
                                例如：<br>
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
                            gr.Markdown("""
                                # Type 2 (Stems)
                                
                                每个文件夹是stem_name。文件夹中包含仅由所需stem组成的wav文件。<br>
                                例如：<br>
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
                            gr.Markdown("""
                                # Type 3 (CSV file)
                                
                                可以提供以下结构的CSV文件（或CSV文件列表）<br>
                                例如：<br>
                                instrum,path<br>
                                vocals,/path/to/dataset/vocals_1.wav<br>
                                vocals,/path/to/dataset2/vocals_v2.wav<br>
                                vocals,/path/to/dataset3/vocals_some.wav<br>
                                ...<br>
                                drums,/path/to/dataset/drums_good.wav<br>
                                ...<br>
                                """)
                            gr.Markdown("""
                                # Type 4 (MUSDB Aligned)
                                
                                与类型1相同，但在训练过程中所有乐器都将来自歌曲的相同位置。<br>
                                例如：<br>
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

                    with gr.Accordion("Step 2: 验证集制作", open=False):
                        gr.Markdown(value="""
                            验证集制作。验证数据集**必须**与上面数据集制作的Type 1(MUSDB)数据集**结构相同**（**无论你使用哪种类型的数据集进行训练**），此外每个文件夹还必须包含每首歌的mixture.wav，mixture.wav是所有stem的总和<br>
                            例如：<br>
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

                    with gr.Accordion("Step 3: 选择并修改修改配置文件", open=False):
                        gr.Markdown(value="""
                            请先明确你想要训练的模型类型，然后选择对应的配置文件进行修改。<br>
                            目前有以下几种模型类型：`mdx23c`, `htdemucs`, `segm_models`, `mel_band_roformer`, `bs_roformer`。<br>
                            确定好模型类型后，你可以前往整合包根目录中的configs_template文件夹下找到对应的配置文件模板。复制一份模板，然后根据你的需求进行修改。修改完成后记下你的配置文件路径，以便后续使用。<br>
                            特别说明：config_musdb18_xxx.yaml是针对MUSDB18数据集的配置文件。<br>
                            """)
                        open_config_template = gr.Button(
                            "打开配置文件模板文件夹", variant="primary")
                        open_config_template.click(open_folder, inputs=gr.Textbox(CONFIG_TEMPLATE_FOLDER, visible=False))
                        gr.Markdown(value="""
                            你可以使用下表根据你的GPU选择用于训练的BS_Roformer模型的batch_size参数。表中提供的批量大小值适用于单个GPU。如果你有多个GPU，则需要将该值乘以GPU的数量。<br>
                            """)
                        roformer_data = {
                            "chunk_size": [131584, 131584, 131584, 131584, 131584, 131584, 263168, 263168, 352800, 352800, 352800, 352800],
                            "dim": [128, 256, 384, 512, 256, 256, 128, 256, 128, 256, 384, 512],
                            "depth": [6, 6, 6, 6, 8, 12, 6, 6, 6, 6, 12, 12],
                            "batch_size (A6000 48GB)": [10, 8, 7, 6, 6, 4, 4, 3, 2, 2, 1, '-'],
                            "batch_size (3090/4090 24GB)": [5, 4, 3, 3, 3, 2, 2, 1, 1, 1, '-', '-'],
                            "batch_size (16GB)": [3, 2, 2, 2, 2, 1, 1, 1, '-', '-', '-', '-']
                        }
                        gr.DataFrame(pd.DataFrame(roformer_data))

                    with gr.Accordion("Step 4: 数据增强", open=False):
                        gr.Markdown(value="""
                            数据增强可以动态更改stem，通过从旧样本创建新样本来增加数据集的大小。现在，数据增强的控制在配置文件中进行。下面是一个包含所有可用数据增强的完整配置示例。你可以将其复制到你的配置文件中以使用数据增强。<br>
                            注意:<br>
                            - 要完全禁用所有数据增强，可以从配置文件中删除augmentations部分或将enable设置为false。<br>
                            - 如果要禁用某些数据增强，只需将其设置为0。<br>
                            - all部分中的数据增强应用于所有stem。<br>
                            - vocals、bass等部分中的数据增强仅应用于相应的stem。你可以为training.instruments中给出的所有stem创建这样的部分。<br>
                            """)
                        augmentations_config = load_augmentations_config()
                        gr.Code(value=augmentations_config, language="yaml")

        with gr.TabItem(label="设置"):
            gr.Markdown(
                value="""选择UVR模型目录：如果你的电脑中有安装UVR5，你不必重新下载一遍UVR5模型，只需在下方“选择UVR模型目录”中选择你的UVR5模型目录，定位到models/VR_Models文件夹。<br>
                例如：E:/Program Files/Ultimate Vocal Remover/models/VR_Models<br>
                点击保存设置或重置设置后，需要重启WebUI以更新。
                """)
            with gr.Row():
                select_uvr_model_dir = gr.Textbox(
                    label="选择UVR模型目录",
                    value=webui_config['settings']['uvr_model_dir'] if webui_config['settings']['uvr_model_dir'] else "pretrain/VR_Models",
                    interactive=True,
                    scale=4
                )
                select_uvr_model_dir_button = gr.Button("选择文件夹", scale=1)
            with gr.Row():
                conform_seetings = gr.Button("保存设置")
                reset_seetings = gr.Button("重置设置")
            gr.Markdown(value="""
                重置WebUI路径记录：将所有输入输出目录重置为默认路径，预设、模型、配置文件以及上面的设置等**不会重置**，无需担心。<br>
                重置WebUI设置后，需要重启WebI。<br>
                ### 点击“重启WebUI”按钮后，会短暂性的失去连接，随后会自动开启一个新网页。
                """)
            reset_all_webui_config = gr.Button("重置WebUI路径记录")
            restart_webui = gr.Button("重启WebUI", variant="primary")
            setting_output_message = gr.Textbox(label="Output Message")

            reset_all_webui_config.click(
                fn=reset_webui_config,
                outputs=setting_output_message
            )
            restart_webui.click(
                fn=webui_restart, outputs=setting_output_message)

            select_uvr_model_dir_button.click(fn=select_folder, outputs=select_uvr_model_dir)
            conform_seetings.click(
                fn=save_settings,
                inputs=[select_uvr_model_dir],
                outputs=setting_output_message
            )
            reset_seetings.click(
                fn=reset_settings,
                inputs=[],
                outputs=setting_output_message
            )

app.launch(inbrowser=True)
