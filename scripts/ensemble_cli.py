import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import time
import shutil
import glob
import traceback
from webui.ensemble import EnsembleFlow
from webui.setup import setup_webui, set_debug
from webui.utils import get_msst_model, load_configs
from utils.constant import *
from utils.logger import get_logger
from utils.ensemble import ensemble_audios
logger = get_logger()


def main(preset_path, ensemble_mode, output_format, store_dir, input_folder, extract_inst):
    preset_data = load_configs(preset_path)

    os.makedirs(store_dir, exist_ok=True)
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    preset = EnsembleFlow(preset_data, force_cpu=False, use_tta=False, logger=logger)
    if preset.total_steps < 2:
        logger.error("Ensemble process requires at least 2 models")
        return
    if not preset.is_exist_models()[0]:
        logger.error(f"Model {preset.is_exist_models()[1]} not found")

    logger.info("Starting ensemble inference process")
    logger.debug(f"presets: {preset.presets}")
    logger.debug(f"total_models: {preset.total_steps}, store_dir: {store_dir}, output_format: {output_format}")

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
            if result[0] == 0:
                logger.error(f"Failed to run VR model {model_name}, error: {result[1]}")
                return
        else:
            model_path, config_path, msst_model_type, _ = get_msst_model(model_name)
            storage = {stem: [temp_store_dir]}
            logger.debug(f"input_folder: {input_folder}, temp_store_dir: {temp_store_dir}, storage: {storage}")
            result = preset.msst_infer(msst_model_type, config_path, model_path, input_folder, storage, "wav")
            if result[0] == 0:
                logger.error(f"Failed to run MSST model {model_name}, error: {result[1]}")
                return

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
    logger.info(f"Total {success_count} audios ensemble successfully, {failed_count} audios failed to ensemble")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Ensemble inference Command Line Interface", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
    parser.add_argument("-p", "--preset_path", type=str, help="Path to the ensemble preset data file (.json). To create a preset file, please refer to the documentation.", required=True)
    parser.add_argument("-m", "--ensemble_mode", type=str, default='avg_wave', choices=ENSEMBLE_MODES, help="Type of ensemble to perform.")
    parser.add_argument("-i", "--input_dir", type=str, default="input", help="Path to the input folder")
    parser.add_argument("-o", "--output_dir", type=str, default="results", help="Path to the output folder")
    parser.add_argument("-f", "--output_format", type=str, default="wav", choices=["wav", "mp3", "flac"], help="Output format of the audio")
    parser.add_argument("--extract_inst", action="store_true", help="Extract instruments by subtracting ensemble result from raw audio")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if not os.path.exists(args.preset_path):
        raise ValueError("Please specify the ensemble preset file")

    if not os.path.exists("configs"):
        shutil.copytree("configs_backup", "configs")
    if not os.path.exists("data"):
        shutil.copytree("data_backup", "data")

    setup_webui() # must be called because we use some functions from webui app
    set_debug(args)

    main(args.preset_path, args.ensemble_mode, args.output_format, args.output_dir, args.input_dir, args.extract_inst)