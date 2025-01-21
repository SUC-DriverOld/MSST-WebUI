__license__= "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr
import pandas as pd
import shutil
import time
import multiprocessing

from utils.constant import *
from utils.logger import get_logger
from webui.utils import (
    i18n, 
    load_configs, 
    save_configs,
    load_vr_model, 
    get_vr_model,
    load_msst_model,
    get_msst_model,
    logger,
    detailed_error
)

def get_presets_list() -> list:
    if os.path.exists(PRESETS):
        presets = [file for file in os.listdir(PRESETS) if file.endswith(".json")]
    else:
        presets = []
    return presets

def preset_backup_list():
    if not os.path.exists(PRESETS_BACKUP):
        return [i18n("暂无备份文件")]
    backup_files = [file for file in os.listdir(PRESETS_BACKUP) if file.endswith(".json")]
    return backup_files

def update_model_stem(model_type, model_name):
    if model_type == "UVR_VR_Models":
        primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
        input_to_next = gr.Radio(
            label=i18n("作为下一模型输入(或结果输出)的音轨"),
            choices=[primary_stem, secondary_stem],
            value=primary_stem,
            interactive=True
        )
        output_to_storage = gr.CheckboxGroup(
            label=i18n("直接保存至输出目录的音轨(可多选)"),
            choices=[i18n("不输出"), primary_stem, secondary_stem],
            interactive=True,
        )
        return input_to_next, output_to_storage
    else:
        _, config_path, _, _ = get_msst_model(model_name)
        stems = load_configs(config_path).training.get("instruments", None)
        input_to_next = gr.Radio(
            label=i18n("作为下一模型输入(或结果输出)的音轨"),
            choices=stems,
            value=stems[0],
            interactive=True
        )
        output_to_storage = gr.CheckboxGroup(
            label=i18n("直接保存至输出目录的音轨(可多选)"),
            choices=[i18n("不输出")] + stems,
            interactive=True,
        )
        return input_to_next, output_to_storage

def add_to_flow_func(model_type, model_name, input_to_next, output_to_storage, df):
    if not model_type or not model_name or not input_to_next:
        return df
    if not output_to_storage or i18n("不输出") in output_to_storage:
        output_to_storage = []
    elif model_type == "UVR_VR_Models":
        primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
        output_to_storage = [stem for stem in output_to_storage if stem in [primary_stem, secondary_stem]]
    else:
        _, config_path, _, _ = get_msst_model(model_name)
        stems = load_configs(config_path).training.get("instruments", None)
        output_to_storage = [stem for stem in output_to_storage if stem in stems]

    new_data = pd.DataFrame({"model_type": [model_type], "model_name": [model_name], "input_to_next": [input_to_next], "output_to_storage": [output_to_storage]})

    if df["model_type"].iloc[0] == "" or df["model_name"].iloc[0] == "" or df["input_to_next"].iloc[0] == "":
        return new_data

    updated_df = pd.concat([df, new_data], ignore_index=True)
    return updated_df

def save_flow_func(preset_name, df):
    if not preset_name:
        output_message = i18n("请填写预设名称")
        return output_message, None, None

    preset_dict = {"version": PRESET_VERSION, "name": preset_name, "flow": df.to_dict(orient="records")}
    os.makedirs(PRESETS, exist_ok=True)
    save_configs(preset_dict, os.path.join(PRESETS, f"{preset_name}.json"))

    output_message = i18n("预设") + preset_name + i18n("保存成功")
    preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
    preset_name_select = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())

    logger.info(f"Save preset: {preset_dict} as {preset_name}")
    return output_message, preset_name_delete, preset_name_select

def reset_flow_func():
    return gr.Dataframe(
        pd.DataFrame({"model_type": [""], "model_name": [""], "input_to_next": [""], "output_to_storage": [""]}),
        interactive=False,
        label=None
    )

def reset_last_func(df):
    if df.shape[0] == 1:
        return reset_flow_func()
    return df.iloc[:-1]

def load_preset(preset_name):
    if preset_name in os.listdir(PRESETS):
        preset_data = load_configs(os.path.join(PRESETS, preset_name))

        version = preset_data.get("version", "Unknown version")
        if version not in SUPPORTED_PRESET_VERSION:
            gr.Warning(i18n("不支持的预设版本: ") + str(version) + i18n(", 请重新制作预设。"))
            logger.error(f"Load preset: {preset_name} failed, unsupported version: {version}, supported version: {SUPPORTED_PRESET_VERSION}")
            return gr.Dataframe(
                pd.DataFrame({"model_type": [i18n("预设版本不支持")], "model_name": [i18n("预设版本不支持")], "input_to_next": [i18n("预设版本不支持")], "output_to_storage": [i18n("预设版本不支持")]}),
                interactive=False,
                label=None
            )

        preset_flow = pd.DataFrame({"model_type": [""], "model_name": [""], "input_to_next": [""], "output_to_storage": [""]})
        for step in preset_data["flow"]:
            preset_flow = add_to_flow_func(
                model_type=step["model_type"],
                model_name=step["model_name"],
                input_to_next=step["input_to_next"],
                output_to_storage=step["output_to_storage"],
                df=preset_flow,
            )
        logger.info(f"Load preset: {preset_name}: {preset_data}")
        return preset_flow

    logger.error(f"Load preset: {preset_name} failed")
    return gr.Dataframe(
        pd.DataFrame({"model_type": [i18n("预设不存在")], "model_name": [i18n("预设不存在")], "input_to_next": [i18n("预设不存在")], "output_to_storage": [i18n("预设不存在")]}),
        interactive=False,
        label=None
    )

def delete_func(preset_name):
    if preset_name in os.listdir(PRESETS):
        select_preset_backup = backup_preset_func(preset_name)
        os.remove(os.path.join(PRESETS, preset_name))
        output_message = i18n("预设") + preset_name + i18n("删除成功")
        preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
        preset_name_select = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
        preset_flow_delete = gr.Dataframe(
            pd.DataFrame({"model_type": [i18n("预设已删除")], "model_name": [i18n("预设已删除")], "input_to_next": [i18n("预设已删除")], "output_to_storage": [i18n("预设已删除")]}),
            interactive=False,
            label=None
        )
        logger.info(f"Delete preset: {preset_name}")
        return output_message, preset_name_delete, preset_name_select, preset_flow_delete, select_preset_backup
    else:
        return i18n("预设不存在"), None, None, None, None

def backup_preset_func(preset_name):
    os.makedirs(PRESETS_BACKUP, exist_ok=True)
    backup_file = f"backup_{preset_name}"
    shutil.copy(os.path.join(PRESETS, preset_name), os.path.join(PRESETS_BACKUP, backup_file))
    logger.info(f"Backup preset: {preset_name} -> {backup_file}")
    return gr.Dropdown(label=i18n("选择需要恢复的预设流程备份"), choices=preset_backup_list(), interactive=True, scale=4)

def restore_preset_func(backup_file):
    backup_file_rename = backup_file
    if backup_file.startswith("backup_"):
        backup_file_rename = backup_file[7:]
    shutil.copy(os.path.join(PRESETS_BACKUP, backup_file), os.path.join(PRESETS, backup_file_rename))
    output_message_manage = i18n("已成功恢复备份") + backup_file
    preset_dropdown = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
    preset_name_delet = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
    preset_flow_delete = pd.DataFrame({"model_type": [i18n("请选择预设")], "model_name": [i18n("请选择预设")], "input_to_next": [i18n("请选择预设")], "output_to_storage": [i18n("请选择预设")]})
    logger.info(f"Restore preset: {backup_file} -> {backup_file_rename}")
    return output_message_manage, preset_dropdown, preset_name_delet, preset_flow_delete


class Presets:
    def __init__(
            self,
            presets,
            force_cpu=False,
            use_tta=False,
            logger=get_logger()
    ):
        self.presets = presets.get("flow", [])
        self.device = "auto" if not force_cpu else "cpu"
        self.force_cpu = force_cpu
        self.use_tta = use_tta
        self.logger = logger
        self.total_steps = len(self.presets)
        self.preset_version = presets.get("version", "Unknown version")
        self.preset_name = presets.get("name", "Unknown name")

        webui_config = load_configs(WEBUI_CONFIG)
        self.debug = webui_config["settings"].get("debug", False)
        self.vr_model_path = webui_config['settings']['uvr_model_dir']
        self.invert_using_spec = webui_config['inference']['vr_invert_spect']
        self.batch_size = int(webui_config['inference']['vr_batch_size'])
        self.window_size = int(webui_config['inference']['vr_window_size'])
        self.aggression = int(webui_config['inference']['vr_aggression'])
        self.enable_post_process = webui_config['inference']['vr_enable_post_process']
        self.post_process_threshold = float(webui_config['inference']['vr_post_process_threshold'])
        self.high_end_process = webui_config['inference']['vr_high_end_process']
        self.wav_bit_depth = webui_config["settings"].get("wav_bit_depth", "FLOAT")
        self.flac_bit_depth = webui_config["settings"].get("flac_bit_depth", "PCM_24")
        self.mp3_bit_rate = webui_config["settings"].get("mp3_bit_rate", "320k")
        
        gpu_id = webui_config["inference"].get("device", None)
        self.gpu_ids = []
        if not self.force_cpu and gpu_id:
            try:
                for gpu in gpu_id:
                    self.gpu_ids.append(int(gpu[:gpu.index(":")]))
            except:
                self.gpu_ids = [0]
        else:
            self.gpu_ids = [0]

    def get_step(self, step):
        return self.presets[step]

    def is_exist_models(self):
        for step in self.presets:
            model_name = step["model_name"]
            if model_name not in load_msst_model() and model_name not in load_vr_model():
                return False, model_name
        return True, None

    def msst_infer(self, model_type, config_path, model_path, input_folder, store_dict, output_format="wav"):
        from webui.msst import run_inference

        result_queue = multiprocessing.Queue()
        msst_inference = multiprocessing.Process(
            target=run_inference,
            args=(
                model_type, config_path, model_path, self.device, self.gpu_ids, output_format,
                self.use_tta, store_dict, self.debug, self.wav_bit_depth, self.flac_bit_depth,
                self.mp3_bit_rate, input_folder, result_queue
            ),
            name="msst_preset_inference"
        )
        msst_inference.start()
        msst_inference.join()

        if result_queue.empty():
            return -1, None
        result = result_queue.get()
        if result[0] == "success":
            return 1, None
        elif result[0] == "error":
            return 0, result[1]

    def vr_infer(self, model_name, input_folder, output_dir, output_format="wav"):
        from webui.vr import run_inference

        model_file = os.path.join(self.vr_model_path, model_name)
        result_queue = multiprocessing.Queue()
        vr_inference = multiprocessing.Process(
            target=run_inference,
            args=(
                self.debug, model_file, output_dir, output_format, self.invert_using_spec, self.force_cpu,
                self.batch_size, self.window_size, self.aggression, self.use_tta, self.enable_post_process,
                self.post_process_threshold, self.high_end_process, self.wav_bit_depth, self.flac_bit_depth,
                self.mp3_bit_rate, input_folder, result_queue
            ),
            name="vr_preset_inference"
        )
        vr_inference.start()
        vr_inference.join()

        if result_queue.empty():
            return -1, None
        result = result_queue.get()
        if result[0] == "success":
            return 1, None
        elif result[0] == "error":
            return 0, result[1]

def preset_inference_audio(input_audio, store_dir, preset, force_cpu, output_format, use_tta, extra_output_dir):
    if not input_audio:
        return i18n("请上传至少一个音频文件!")
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)
    os.makedirs(os.path.join(TEMP_PATH, "step_0_output"))

    for audio in input_audio:
        shutil.copy(audio, os.path.join(TEMP_PATH, "step_0_output"))
    input_folder = os.path.join(TEMP_PATH, "step_0_output")
    msg = preset_inference(input_folder, store_dir, preset, force_cpu, output_format, use_tta, extra_output_dir, is_audio=True)
    return msg

def preset_inference(input_folder, store_dir, preset_name, force_cpu, output_format, use_tta, extra_output_dir: bool, is_audio=False):
    if preset_name not in os.listdir(PRESETS):
        return i18n("预设") + preset_name + i18n("不存在")

    preset_data = load_configs(os.path.join(PRESETS, preset_name))
    preset_version = preset_data.get("version", "Unknown version")
    if preset_version not in SUPPORTED_PRESET_VERSION:
        logger.error(f"Unsupported preset version: {preset_version}, supported version: {SUPPORTED_PRESET_VERSION}")
        return i18n("不支持的预设版本: ") + preset_version + i18n(", 请重新制作预设。")

    config = load_configs(WEBUI_CONFIG)
    config['inference']['preset'] = preset_name
    config['inference']['force_cpu'] = force_cpu
    config['inference']['output_format'] = output_format
    config['inference']['preset_use_tta'] = use_tta
    config['inference']['store_dir'] = store_dir
    config['inference']['extra_output_dir'] = extra_output_dir
    if not is_audio:
        config['inference']['input_dir'] = input_folder
    save_configs(config, WEBUI_CONFIG)

    os.makedirs(store_dir, exist_ok=True)

    direct_output = store_dir
    if extra_output_dir:
        os.makedirs(os.path.join(store_dir, "extra_output"), exist_ok=True)
        direct_output = os.path.join(store_dir, "extra_output")

    input_to_use = input_folder
    if os.path.exists(TEMP_PATH) and not is_audio:
        shutil.rmtree(TEMP_PATH)
    tmp_store_dir = os.path.join(TEMP_PATH, "step_1_output")

    preset = Presets(preset_data, force_cpu, use_tta, logger)

    logger.info(f"Starting preset inference process, use presets: {preset_name}")
    logger.debug(f"presets: {preset.presets}")
    logger.debug(f"total_steps: {preset.total_steps}, force_cpu: {force_cpu}, use_tta: {use_tta}, store_dir: {store_dir}, extra_output_dir: {extra_output_dir}, output_format: {output_format}")

    if not preset.is_exist_models()[0]:
        return i18n("模型") + preset.is_exist_models()[1] + i18n("不存在")

    start_time = time.time()
    current_step = 0

    for step in range(preset.total_steps):
        if current_step == 0:
            input_to_use = input_folder
        if preset.total_steps - 1 > current_step > 0:
            if input_to_use != input_folder:
                shutil.rmtree(input_to_use)
            input_to_use = tmp_store_dir
            tmp_store_dir = os.path.join(TEMP_PATH, f"step_{current_step + 1}_output")
        if current_step == preset.total_steps - 1:
            input_to_use = tmp_store_dir
            tmp_store_dir = store_dir
        if preset.total_steps == 1:
            input_to_use = input_folder
            tmp_store_dir = store_dir

        data = preset.get_step(step)
        model_type = data["model_type"]
        model_name = data["model_name"]
        input_to_next = data["input_to_next"]
        output_to_storage = data["output_to_storage"]

        logger.info(f"\033[33mStep {current_step + 1}: Running inference using {model_name}\033[0m")

        if model_type == "UVR_VR_Models":
            primary_stem, secondary_stem, _, _= get_vr_model(model_name)
            storage = {primary_stem:[], secondary_stem:[]}
            storage[input_to_next].append(tmp_store_dir)
            for stem in output_to_storage:
                storage[stem].append(direct_output)

            logger.debug(f"input_to_next: {input_to_next}, output_to_storage: {output_to_storage}, storage: {storage}")
            result = preset.vr_infer(model_name, input_to_use, storage, output_format)
            if result[0] == -1:
                return i18n("用户强制终止")
            elif result[0] == 0:
                return i18n("处理失败: ") + detailed_error(result[1])
        else:
            model_path, config_path, msst_model_type, _ = get_msst_model(model_name)
            stems = load_configs(config_path).training.get("instruments", [])
            storage = {stem:[] for stem in stems}
            storage[input_to_next].append(tmp_store_dir)
            for stem in output_to_storage:
                storage[stem].append(direct_output)

            logger.debug(f"input_to_next: {input_to_next}, output_to_storage: {output_to_storage}, storage: {storage}")
            result = preset.msst_infer(msst_model_type, config_path, model_path, input_to_use, storage, output_format)
            if result[0] == -1:
                return i18n("用户强制终止")
            elif result[0] == 0:
                return i18n("处理失败: ") + detailed_error(result[1])
        current_step += 1

    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    logger.info(f"\033[33mPreset: {preset_name} inference process completed, results saved to {store_dir}, "
                f"time cost: {round(time.time() - start_time, 2)}s\033[0m")
    return i18n("处理完成, 结果已保存至: ") + store_dir + i18n(", 耗时: ") + \
        str(round(time.time() - start_time, 2)) + "s"

def stop_preset():
    for process in multiprocessing.active_children():
        if process.name in ["msst_preset_inference", "vr_preset_inference"]:
            process.terminate()
            process.join()
            logger.info(f"Inference process stopped, PID: {process.pid}")