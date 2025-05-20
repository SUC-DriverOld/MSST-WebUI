import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import time
import shutil
from inference.preset_infer import EnsembleInfer
from webui.setup import setup_webui, set_debug
from webui.utils import load_configs
from utils.constant import *
from utils.logger import get_logger

logger = get_logger()


def main(preset_path, ensemble_mode, output_format, store_dir, input_folder, extract_inst):
	preset_data = load_configs(preset_path)

	os.makedirs(store_dir, exist_ok=True)
	if os.path.exists(TEMP_PATH):
		shutil.rmtree(TEMP_PATH)

	start_time = time.time()
	logger.info("Starting ensemble inference process")
	preset = EnsembleInfer(preset_data, force_cpu=False, use_tta=False, logger=logger, callback=None)
	logger.debug(f"presets: {preset.presets}")
	logger.debug(f"total_models: {preset.total_steps}, store_dir: {store_dir}, output_format: {output_format}")
	preset.process_folder(input_folder)
	results = preset.ensemble(input_folder, store_dir, ensemble_mode, output_format, extract_inst)
	logger.info(f"Ensemble inference completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
	import multiprocessing

	multiprocessing.set_start_method("spawn", force=True)

	parser = argparse.ArgumentParser(description="Ensemble inference Command Line Interface", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
	parser.add_argument("-p", "--preset_path", type=str, help="Path to the ensemble preset data file (.json). To create a preset file, please refer to the documentation.", required=True)
	parser.add_argument("-m", "--ensemble_mode", type=str, default="avg_wave", choices=ENSEMBLE_MODES, help="Type of ensemble to perform.")
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

	setup_webui()  # must be called because we use some functions from webui app
	set_debug(args)

	main(args.preset_path, args.ensemble_mode, args.output_format, args.output_dir, args.input_dir, args.extract_inst)
