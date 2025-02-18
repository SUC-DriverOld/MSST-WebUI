import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import time
import shutil
from webui.preset import Presets
from webui.utils import load_configs, get_vr_model, get_msst_model
from webui.setup import setup_webui, set_debug
from utils.constant import *
from utils.logger import get_logger
logger = get_logger()


def main(input_folder, store_dir, preset_path, output_format, extra_output_dir: bool=False):
    preset_data = load_configs(preset_path)
    preset_version = preset_data.get("version", "Unknown version")
    if preset_version not in SUPPORTED_PRESET_VERSION:
        logger.error(f"Unsupported preset version: {preset_version}, supported version: {SUPPORTED_PRESET_VERSION}")

    os.makedirs(store_dir, exist_ok=True)

    direct_output = store_dir
    if extra_output_dir:
        os.makedirs(os.path.join(store_dir, "extra_output"), exist_ok=True)
        direct_output = os.path.join(store_dir, "extra_output")

    input_to_use = input_folder
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)
    tmp_store_dir = os.path.join(TEMP_PATH, "step_1_output")

    preset = Presets(preset_data, force_cpu=False, use_tta=False, logger=logger)

    logger.info(f"Starting preset inference process, use presets: {preset_path}")
    logger.debug(f"presets: {preset.presets}")
    logger.debug(f"total_steps: {preset.total_steps}, store_dir: {store_dir}, output_format: {output_format}")

    if not preset.is_exist_models()[0]:
        logger.error(f"Model {preset.is_exist_models()[1]} not found")

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
            if result[0] == 0:
                logger.error(f"Failed to run VR model {model_name}, error: {result[1]}")
                return
        else:
            model_path, config_path, msst_model_type, _ = get_msst_model(model_name)
            stems = load_configs(config_path).training.get("instruments", [])
            storage = {stem:[] for stem in stems}
            storage[input_to_next].append(tmp_store_dir)
            for stem in output_to_storage:
                storage[stem].append(direct_output)

            logger.debug(f"input_to_next: {input_to_next}, output_to_storage: {output_to_storage}, storage: {storage}")
            result = preset.msst_infer(msst_model_type, config_path, model_path, input_to_use, storage, output_format)
            if result[0] == 0:
                logger.error(f"Failed to run MSST model {model_name}, error: {result[1]}")
                return
        current_step += 1

    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    logger.info(f"\033[33mPreset: {preset_path} inference process completed, results saved to {store_dir}, "
                f"time cost: {round(time.time() - start_time, 2)}s\033[0m")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Preset inference Command Line Interface", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
    parser.add_argument("-p", "--preset_path", type=str, help="Path to the preset file (*.json). To create a preset file, please refer to the documentation or use WebUI to create one.", required=True)
    parser.add_argument("-i", "--input_dir", type=str, default="input", help="Path to the input folder")
    parser.add_argument("-o", "--output_dir", type=str, default="results", help="Path to the output folder")
    parser.add_argument("-f", "--output_format", type=str, default="wav", choices=["wav", "mp3", "flac"], help="Output format of the audio")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--extra_output_dir", action="store_true", help="Enable extra output directory")
    args = parser.parse_args()

    if not os.path.exists(args.preset_path):
        raise ValueError("Please specify the preset file")

    if not os.path.exists("configs"):
        shutil.copytree("configs_backup", "configs")
    if not os.path.exists("data"):
        shutil.copytree("data_backup", "data")

    setup_webui() # must be called because we use some functions from webui app
    set_debug(args)

    main(args.input_dir, args.output_dir, args.preset_path, args.output_format, args.extra_output_dir)
