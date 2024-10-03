import gradio as gr
import pandas as pd
import shutil
import time
import rich
from datetime import datetime

from tools.webUI.constant import *
from tools.webUI.utils import (
    i18n, 
    load_configs, 
    save_configs, 
    load_msst_model, 
    load_vr_model, 
    get_vr_model, 
    get_stop_infer_flow, 
    change_stop_infer_flow,
    load_selected_model,
    get_msst_model
)
from inference.msst import run_inference
from inference.vr import vr_inference

def preset_backup_list():
    backup_dir = BACKUP
    if not os.path.exists(backup_dir):
        return [i18n("暂无备份文件")]

    backup_files = []
    for file in os.listdir(backup_dir):
        if file.startswith("preset_backup_") and file.endswith(".json"):
            backup_files.append(file)
    return backup_files

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
        _, config_path, _, _ = get_msst_model(model_name)
        config = load_configs(config_path)
        stem = ["All_stems"] if config.training.get("target_instrument", None) is None else [config.training.get("target_instrument")]
        return gr.Dropdown(label=i18n("输出音轨"), choices=stem, value=stem[0], interactive=False)

def add_to_flow_func(model_type, model_name, stem, secondary_output, df):
    if not model_type or not model_name or (model_type == "UVR_VR_Models" and not stem):
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
    os.makedirs(backup_dir, exist_ok=True)
    backup_file = f"preset_backup_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    preset_data = load_configs(PRESETS)
    save_configs(preset_data, os.path.join(backup_dir, backup_file))
    msg = i18n("已成功备份至") + backup_file
    select_preset_backup = gr.Dropdown(label=i18n("选择需要恢复的预设流程备份"), choices=preset_backup_list(), interactive=True, scale=4)
    return msg, select_preset_backup

def run_single_inference_flow(input_audio, store_dir, preset_name, force_cpu, output_format_flow):
    if not input_audio:
        return i18n("请上传至少一个音频文件!")
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    os.makedirs(os.path.join(TEMP_PATH, "inferflow_step0_output"))

    for audio in input_audio:
        shutil.copy(audio, os.path.join(TEMP_PATH, "inferflow_step0_output"))

    input_folder = os.path.join(TEMP_PATH, "inferflow_step0_output")
    msg = run_inference_flow(input_folder, store_dir, preset_name, force_cpu, output_format_flow, isSingle=True)
    return msg

def run_inference_flow(input_folder, store_dir, preset_name, force_cpu, output_format_flow, isSingle=False):
    change_stop_infer_flow()
    start_time = time.time()
    preset_data = load_configs(PRESETS)

    if not preset_name in preset_data.keys():
        return i18n("预设") + preset_name + i18n("不存在")

    config = load_configs(WEBUI_CONFIG)
    config['inference']['preset'] = preset_name
    config['inference']['force_cpu'] = force_cpu
    config['inference']['output_format_flow'] = output_format_flow
    config['inference']['store_dir_flow'] = store_dir

    if not isSingle:
        config['inference']['input_folder_flow'] = input_folder
    else: 
        pass

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
    console = rich.console.Console()

    for step in model_list.keys():
        if get_stop_infer_flow():
            change_stop_infer_flow()
            break

        if i == 0:
            input_to_use = input_folder

        if i < len(model_list.keys()) - 1 and i > 0:
            if input_to_use != input_folder:
                shutil.rmtree(input_to_use)
            input_to_use = tmp_store_dir
            tmp_store_dir = f"{TEMP_PATH}/inferflow_step{i+1}_output"

        if i == len(model_list.keys()) - 1:
            input_to_use = tmp_store_dir
            tmp_store_dir = store_dir

        if len(model_list.keys()) == 1:
            input_to_use = input_folder
            tmp_store_dir = store_dir

        model_name = model_list[step]["model_name"]
        console.print(f"[yellow]Step {i+1}: Running inference using {model_name}", style="yellow", justify='center')

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
            device = config['inference']['device']

            if device is None or len(device) == 0:
                device = ["0"]

            use_tta = config['inference']['use_tta']
            instrumental_only = False

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

            run_inference(model_name, input_to_use, tmp_store_dir, extract_instrumental, device, output_format_flow, force_cpu, use_tta, instrumental_only, extra_store_dir)

        i += 1

    shutil.rmtree(TEMP_PATH)
    finish_time = time.time()
    elapsed_time = finish_time - start_time
    console.rule(f"[yellow]Finished runing {preset_name}! Costs {elapsed_time:.2f}s", style="yellow")

    return i18n("运行完成, 耗时: ") + str(round(elapsed_time, 2)) + "s"