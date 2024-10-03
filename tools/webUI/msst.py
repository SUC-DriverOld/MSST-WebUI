import os
import shutil
import threading
import time
import gradio as gr

from tools.webUI.constant import *
from tools.webUI.utils import i18n, load_configs, save_configs, run_command, load_selected_model
from tools.webUI.init import get_msst_model

def save_config(selected_model, batch_size, dim_t, num_overlap, normalize):
    _, config_path, _, _ = get_msst_model(selected_model)
    config = load_configs(config_path)

    if config.inference.get('batch_size'):
        config.inference['batch_size'] = int(batch_size)
    if config.inference.get('dim_t'):
        config.inference['dim_t'] = int(dim_t)
    if config.inference.get('num_overlap'):
        config.inference['num_overlap'] = int(num_overlap)
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

def update_inference_settings(selected_model):
    batch_size = gr.Number(label="batch_size", value=i18n("该模型不支持修改此值"), interactive=False)
    dim_t = gr.Number(label="dim_t", value=i18n("该模型不支持修改此值"), interactive=False)
    num_overlap = gr.Number(label="num_overlap", value=i18n("该模型不支持修改此值"), interactive=False)
    normalize = gr.Checkbox(label=i18n("normalize (该模型不支持修改此值) "), value=False, interactive=False)
    extract_instrumental = gr.Checkbox(label=i18n("同时输出次级音轨"), interactive=True)
    instrumental_only = gr.Checkbox(label=i18n("仅输出次级音轨"), interactive=True)

    if selected_model and selected_model !="":
        _, config_path, _, _ = get_msst_model(selected_model)
        config = load_configs(config_path)

        if config.inference.get('batch_size'):
            batch_size = gr.Number(label="batch_size", value=str(config.inference.get('batch_size')), interactive=True)
        if config.inference.get('dim_t'):
            dim_t = gr.Number(label="dim_t", value=str(config.inference.get('dim_t')), interactive=True)
        if config.inference.get('num_overlap'):
            num_overlap = gr.Number(label="num_overlap", value=str(config.inference.get('num_overlap')), interactive=True)
        if config.inference.get('normalize'):
            normalize = gr.Checkbox(label="normalize", value=config.inference.get('normalize'), interactive=True)
        target_inst = config.training.get('target_instrument', None)

        if target_inst is None:
            extract_instrumental = gr.Checkbox(label=i18n("此模型默认输出所有音轨"), interactive=False)
            instrumental_only = gr.Checkbox(label=i18n("此模型默认输出所有音轨"), interactive=False)

    return batch_size, dim_t, num_overlap, normalize, extract_instrumental, instrumental_only

def update_selected_model(model_type):
    webui_config = load_configs(WEBUI_CONFIG)
    webui_config["inference"]["model_type"] = model_type
    save_configs(webui_config, WEBUI_CONFIG)
    return gr.Dropdown(label=i18n("选择模型"), choices=load_selected_model(), value=None, interactive=True, scale=4)

def save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta, instrumental_only):
    config = load_configs(WEBUI_CONFIG)
    config['inference']['selected_model'] = selected_model
    config['inference']['device'] = gpu_id
    config['inference']['output_format'] = output_format
    config['inference']['force_cpu'] = force_cpu
    config['inference']['extract_instrumental'] = extract_instrumental
    config['inference']['instrumental_only'] = instrumental_only
    config['inference']['use_tta'] = use_tta
    config['inference']['store_dir'] = store_dir
    config['inference']['multiple_audio_input'] = input_folder
    save_configs(config, WEBUI_CONFIG)

def run_inference_single(selected_model, input_audio, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta, instrumental_only):
    input_folder = None

    if not input_audio:
        return i18n("请上传至少一个音频文件!")
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    os.makedirs(TEMP_PATH)

    for audio in input_audio:
        shutil.copy(audio, TEMP_PATH)
    input_path = TEMP_PATH
    start_time = time.time()

    save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta, instrumental_only)

    run_inference(selected_model, input_path, store_dir,extract_instrumental, gpu_id, output_format, force_cpu, use_tta, instrumental_only)

    shutil.rmtree(TEMP_PATH)
    return i18n("运行完成, 耗时: ") + str(round(time.time() - start_time, 2)) + "s"

def run_multi_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta, instrumental_only):
    start_time = time.time()

    save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta, instrumental_only)

    run_inference(selected_model, input_folder, store_dir,extract_instrumental, gpu_id, output_format, force_cpu, use_tta, instrumental_only)

    return i18n("运行完成, 耗时: ") + str(round(time.time() - start_time, 2)) + "s"

def run_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta, instrumental_only, extra_store_dir=None):
    gpu_ids = []

    if not force_cpu:
        if len(gpu_id) == 0:
            raise gr.Error(i18n("请选择GPU"))
        try:
            for gpu in gpu_id:
                gpu_ids.append(gpu[:gpu.index(":")])
        except:
            gpu_ids = ["0"]

    if selected_model == "":
        raise gr.Error(i18n("请选择模型"))
    if input_folder == "":
        raise gr.Error(i18n("请选择输入目录"))

    os.makedirs(store_dir, exist_ok=True)

    if extra_store_dir:
        os.makedirs(extra_store_dir, exist_ok=True)

    start_check_point, config_path, model_type, _ = get_msst_model(selected_model)
    gpu_ids = " ".join(gpu_ids) if not force_cpu else "0"
    extract_instrumental_option = "--extract_instrumental" if extract_instrumental else ""
    force_cpu_option = "--force_cpu" if force_cpu else ""
    use_tta_option = "--use_tta" if use_tta else ""
    instrumental_only = "--instrumental_only" if instrumental_only else ""
    extra_store_dir = f"--extra_store_dir \"{extra_store_dir}\"" if extra_store_dir else ""

    command = f"{PYTHON} inference/msst_cli.py --model_type {model_type} --config_path \"{config_path}\" --start_check_point \"{start_check_point}\" --input_folder \"{input_folder}\" --store_dir \"{store_dir}\" --device_ids {gpu_ids} --output_format {output_format} {extract_instrumental_option} {instrumental_only} {force_cpu_option} {use_tta_option} {extra_store_dir}"

    msst_inference = threading.Thread(target=run_command, args=(command,), name="msst_inference")
    msst_inference.start()
    msst_inference.join()