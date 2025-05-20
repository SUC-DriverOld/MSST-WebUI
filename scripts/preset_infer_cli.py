import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import time
import shutil
from inference.preset_infer import PresetInfer
from webui.utils import load_configs
from webui.setup import setup_webui, set_debug
from utils.constant import *
from utils.logger import get_logger

logger = get_logger()


def main(input_folder, store_dir, preset_path, output_format, extra_output_dir: bool = False):
	preset_data = load_configs(preset_path)
	preset_version = preset_data.get("version", "Unknown version")
	if preset_version not in SUPPORTED_PRESET_VERSION:
		logger.error(f"Unsupported preset version: {preset_version}, supported version: {SUPPORTED_PRESET_VERSION}")

	os.makedirs(store_dir, exist_ok=True)

	if os.path.exists(TEMP_PATH):
		shutil.rmtree(TEMP_PATH)

	start_time = time.time()
	logger.info(f"Starting preset inference process, use presets: {preset_path}")
	preset = PresetInfer(preset_data, force_cpu=False, use_tta=False, logger=logger, callback=None)
	logger.debug(f"presets: {preset.presets}")
	logger.debug(f"total_steps: {preset.total_steps}, store_dir: {store_dir}, extra_output_dir: {extra_output_dir}, output_format: {output_format}")
	preset.process_folder(input_folder, store_dir, output_format, extra_output_dir)
	logger.info(f"Preset inference completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
	import multiprocessing

	multiprocessing.set_start_method("spawn", force=True)

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

	setup_webui()  # must be called because we use some functions from webui app
	set_debug(args)

	main(args.input_dir, args.output_dir, args.preset_path, args.output_format, args.extra_output_dir)
