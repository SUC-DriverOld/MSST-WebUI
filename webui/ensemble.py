__license__= "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr
import pandas as pd
import traceback
import soundfile as sf
import numpy as np
import shutil
import time
import multiprocessing
import glob

from pydub import AudioSegment
from utils.constant import *
from utils.logger import get_logger
from utils.ensemble import ensemble_audios
from webui.preset import Presets
from webui.utils import (
    i18n, 
    load_configs, 
    save_configs,
    get_vr_model,
    get_msst_model,
    logger
)

class EnsembleFlow(Presets):
    def __init__(self, presets={}, force_cpu=False, use_tta=False, logger=get_logger()):
        super().__init__(presets, force_cpu, use_tta, logger)
        self.presets = presets.get("flow", [])

    def save_audio(self, audio, sr, output_format, file_name, store_dir):
        if output_format.lower() == 'flac':
            file = os.path.join(store_dir, file_name + '.flac')
            sf.write(file, audio, sr, subtype=self.flac_bit_depth)
        elif output_format.lower() == 'mp3':
            file = os.path.join(store_dir, file_name + '.mp3')
            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=sr,
                sample_width=audio.dtype.itemsize,
                channels=2
                )
            audio_segment.export(file, format='mp3', bitrate=self.mp3_bit_rate)
        else:
            file = os.path.join(store_dir, file_name + '.wav')
            sf.write(file, audio, sr, subtype=self.wav_bit_depth)
        return file


def update_model_stem(model_type, model_name):
    if model_type == "UVR_VR_Models":
        primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
        output_stems = gr.Radio(
            label=i18n("输出音轨"),
            choices=[primary_stem, secondary_stem],
            value=primary_stem,
            interactive=True
        )
        return output_stems
    else:
        _, config_path, _, _ = get_msst_model(model_name)
        stems = load_configs(config_path).training.get("instruments", None)
        output_stems = gr.Radio(
            label=i18n("输出音轨"),
            choices=stems,
            value=stems[0],
            interactive=True
        )
        return output_stems

def add_to_ensemble_flow(model_type, model_name, stem_weight, stem, df):
    if not model_type or not model_name or not stem_weight or not stem:
        return df
    new_data = pd.DataFrame({"model_type": [model_type], "model_name": [model_name], "stem": [stem], "weight": [stem_weight]})
    if df["model_type"].iloc[0] == "" or df["model_name"].iloc[0] == "" or df["stem"].iloc[0] == "" or df["weight"].iloc[0] == "":
        return new_data
    updated_df = pd.concat([df, new_data], ignore_index=True)
    return updated_df

def reset_flow_func():
    return gr.Dataframe(
        pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""], "weight": [""]}),
        interactive=False,
        label=None
    )

def reset_last_func(df):
    if df.shape[0] == 1:
        return reset_flow_func()
    return df.iloc[:-1]

def save_ensemble_preset_func(df):
    if df.shape[0] < 2:
        raise gr.Error(i18n("请至少添加2个模型到合奏流程"))
    config = load_configs(WEBUI_CONFIG)
    config['inference']['ensemble_preset'] = {'flow': df.to_dict(orient="records")}
    save_configs(config, WEBUI_CONFIG)
    logger.info(f"Ensemble flow saved: {df.to_dict(orient='records')}")
    gr.Info(i18n("合奏流程已保存"))

def load_ensemble():
    flow = pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""], "weight": [""]})
    try:
        config = load_configs(WEBUI_CONFIG)
        data = config['inference']['ensemble_preset']
        if not data:
            return flow
        for step in data["flow"]:
            flow = add_to_ensemble_flow(
                model_type=step["model_type"],
                model_name=step["model_name"],
                stem_weight=step["weight"],
                stem=step["stem"],
                df=flow,
            )
        return flow
    except:
        return flow

def inference_audio_func(ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_audio, extract_inst):
    if not input_audio:
        return i18n("请上传至少一个音频文件!")
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)
    os.makedirs(os.path.join(TEMP_PATH, "ensemble_raw"))

    for audio in input_audio:
        shutil.copy(audio, os.path.join(TEMP_PATH, "ensemble_raw"))
    input_folder = os.path.join(TEMP_PATH, "ensemble_raw")
    msg = inference_folder_func(ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_folder, extract_inst, is_audio=True)
    return msg

def inference_folder_func(ensemble_mode, output_format, force_cpu, use_tta, store_dir, input_folder, extract_inst, is_audio=False):
    config = load_configs(WEBUI_CONFIG)
    preset_data = config['inference']['ensemble_preset']
    if not preset_data:
        return i18n("请先创建合奏流程")

    config['inference']['force_cpu'] = force_cpu
    config['inference']['output_format'] = output_format
    config['inference']['store_dir'] = store_dir
    config['inference']['ensemble_use_tta'] = use_tta
    config['inference']['ensemble_type'] = ensemble_mode
    config['inference']['ensemble_extract_inst'] = extract_inst

    if not is_audio:
        config['inference']['input_dir'] = input_folder
    save_configs(config, WEBUI_CONFIG)
    os.makedirs(store_dir, exist_ok=True)
    if os.path.exists(TEMP_PATH) and not is_audio:
        shutil.rmtree(TEMP_PATH)

    preset = EnsembleFlow(preset_data, force_cpu, use_tta, logger)
    if preset.total_steps < 2:
        return i18n("请至少添加2个模型到合奏流程")
    if not preset.is_exist_models()[0]:
        return i18n("模型") + preset.is_exist_models()[1] + i18n("不存在")

    logger.info("Starting ensemble inference process")
    logger.debug(f"presets: {preset.presets}")
    logger.debug(f"total_models: {preset.total_steps}, force_cpu: {force_cpu}, use_tta: {use_tta}, store_dir: {store_dir}, output_format: {output_format}")

    start_time = time.time()
    ensemble_data = {}
    for data in preset.presets:
        model_type = data["model_type"]
        model_name = data["model_name"]
        stem = data["stem"]
        temp_store_dir = os.path.join(TEMP_PATH, model_name)
        ensemble_data[model_name] = {"store_dir": temp_store_dir, "weight": float(data["weight"])}

        logger.info(f"\033[33mRunning inference using {model_name}\033[0m")

        if model_type == "UVR_VR_Models":
            storage = {stem: [temp_store_dir]}
            logger.debug(f"input_folder: {input_folder}, temp_store_dir: {temp_store_dir}, storage: {storage}")
            result = preset.vr_infer(model_name, input_folder, storage, "wav")
            if result[0] == -1:
                return i18n("用户强制终止")
            elif result[0] == 0:
                return i18n("处理失败: ") + result[1]
        else:
            model_path, config_path, msst_model_type, _ = get_msst_model(model_name)
            storage = {stem: [temp_store_dir]}
            logger.debug(f"input_folder: {input_folder}, temp_store_dir: {temp_store_dir}, storage: {storage}")
            result = preset.msst_infer(msst_model_type, config_path, model_path, input_folder, storage, "wav")
            if result[0] == -1:
                return i18n("用户强制终止")
            elif result[0] == 0:
                return i18n("处理失败: ") + result[1]

    logger.info(f"\033[33mInference process completed, time cost: {round(time.time() - start_time, 2)}s, starting ensemble...\033[0m")

    success_count = 0
    failed_count = 0

    for audio in os.listdir(input_folder):
        base_name = os.path.splitext(audio)[0]
        ensemble_audio = []
        ensemble_weights = []
        try:
            for model_name in ensemble_data.keys():
                audio_folder = ensemble_data[model_name]["store_dir"]
                audio_file = glob.glob(os.path.join(audio_folder, f"{base_name}*"))[0]
                ensemble_audio.append(audio_file)
                ensemble_weights.append(ensemble_data[model_name]["weight"])

            logger.debug(f"ensemble_audio: {ensemble_audio}, ensemble_weights: {ensemble_weights}")
            res, sr = ensemble_audios(ensemble_audio, ensemble_mode, ensemble_weights)
            save_filename = f"{base_name}_ensemble_{ensemble_mode}"
            preset.save_audio(res, sr, output_format, save_filename, store_dir)

            if extract_inst:
                import librosa
                logger.debug(f"User choose to extract other instruments")
                raw, _ = librosa.load(os.path.join(input_folder, audio), sr=sr, mono=False)
                res = res.T

                if raw.shape[-1] != res.shape[-1]:
                    logger.warning(f"Extracted audio shape: {res.shape} is not equal to raw audio shape: {raw.shape}, matching min length")
                    min_length = min(raw.shape[-1], res.shape[-1])
                    raw = raw[..., :min_length]
                    res = res[..., :min_length]

                result = raw - res
                logger.debug(f"Extracted audio shape: {result.shape}")
                save_inst = f"{base_name}_ensemble_{ensemble_mode}_other"
                preset.save_audio(result.T, sr, output_format, save_inst, store_dir)

            success_count += 1

        except Exception as e:
            logger.error(f"Fail to ensemble audio: {audio}. Error: {e}\n{traceback.format_exc()}")
            failed_count += 1
            continue

    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    logger.info(f"Ensemble process completed, saved to: {store_dir}, total time cost: {round(time.time() - start_time, 2)}s")
    return i18n("处理完成, 成功: ") + str(success_count) + i18n("个文件, 失败: ") + str(failed_count) + i18n("个文件") + i18n(", 结果已保存至: ") + store_dir + i18n(", 耗时: ") + str(round(time.time() - start_time, 2)) + "s"

def stop_ensemble_func():
    for process in multiprocessing.active_children():
        if process.name in ["msst_preset_inference", "vr_preset_inference"]:
            process.terminate()
            process.join()
            logger.info(f"Inference process stopped, PID: {process.pid}")

def ensemble_files(files, ensemble_mode, weights, output_path, output_format):
    if len(files) < 2:
        return i18n("请上传至少2个文件")
    if len(files) != len(weights.split()):
        return i18n("上传的文件数目与权重数目不匹配")

    config = load_configs(WEBUI_CONFIG)
    config['inference']['ensemble_type'] = ensemble_mode
    config['inference']['store_dir'] = output_path
    save_configs(config, WEBUI_CONFIG)

    os.makedirs(output_path, exist_ok=True)
    weights = [float(w) for w in weights.split()]
    filename = f"ensemble_{ensemble_mode}_{len(files)}_songs"
    try:
        ensemble = EnsembleFlow()
        res, sr = ensemble_audios(files, ensemble_mode, weights)
        file = ensemble.save_audio(res, sr, output_format, filename, output_path)
        logger.info(f"Ensemble files completed, saved to: {file}")
        return i18n("处理完成, 文件已保存为: ") + file
    except Exception as e:
        logger.error(f"Fail to ensemble files. Error: {e}\n{traceback.format_exc()}")
        return i18n("处理失败!") + str(e)