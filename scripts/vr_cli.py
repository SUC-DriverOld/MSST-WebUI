import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import warnings
import logging
from time import time
from inference.vr_infer import VRSeparator
from utils.logger import get_logger


def vr_inference(args):
	logger = get_logger(console_level=logging.INFO)

	if not args.debug:
		warnings.filterwarnings("ignore", category=UserWarning)

	start_time = time()

	separator = VRSeparator(
		logger=logger,
		debug=args.debug,
		model_file=args.model_path,
		output_dir=args.output_folder,
		output_format=args.output_format,
		use_cpu=args.use_cpu,
		vr_params={
			"batch_size": args.batch_size,
			"window_size": args.window_size,
			"aggression": args.aggression,
			"enable_tta": args.enable_tta,
			"enable_post_process": args.enable_post_process,
			"post_process_threshold": args.post_process_threshold,
			"high_end_process": args.high_end_process,
		},
		audio_params={"wav_bit_depth": args.wav_bit_depth, "flac_bit_depth": args.flac_bit_depth, "mp3_bit_rate": args.mp3_bit_rate},
	)
	success_files = separator.process_folder(args.input_folder)
	separator.del_cache()
	logger.info(f"Successfully separated files: {success_files}, total time: {time() - start_time:.2f} seconds.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Vocal Remover Command Line Interface", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))

	parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging (default: %(default)s). Example: --debug")
	parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU for inference (default: %(default)s). Example: --use_cpu")

	io_params = parser.add_argument_group("Separation I/O Params")
	io_params.add_argument("-i", "--input_folder", type=str, default="input", help="Folder with mixtures to process.")
	io_params.add_argument(
		"-o", "--output_folder", type=str, default="results", help="Folder to store separated files. Only can be str when using cli (default: %(default)s). Example: --output_folder=results"
	)
	io_params.add_argument("--output_format", choices=["wav", "flac", "mp3"], default="wav", help="Output format for separated files (default: %(default)s). Example: --output_format=wav")

	vr_params = parser.add_argument_group("VR Architecture Parameters")
	vr_params.add_argument("-m", "--model_path", type=str, help="Path to model checkpoint.", required=True)
	vr_params.add_argument(
		"--batch_size", type=int, default=2, help="Number of batches to process at a time. higher = more RAM, slightly faster processing (default: %(default)s). Example: --batch_size=16"
	)
	vr_params.add_argument(
		"--window_size", type=int, default=512, help="Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: %(default)s). Example: --window_size=320"
	)
	vr_params.add_argument(
		"--aggression", type=int, default=5, help="Intensity of primary stem extraction, -100 - 100. typically 5 for vocals & instrumentals (default: %(default)s). Example: --aggression=2"
	)
	vr_params.add_argument("--enable_tta", action="store_true", help="Enable Test-Time-Augmentation, slow but improves quality (default: %(default)s). Example: --enable_tta")
	vr_params.add_argument("--high_end_process", action="store_true", help="Mirror the missing frequency range of the output (default: %(default)s). Example: --high_end_process")
	vr_params.add_argument(
		"--enable_post_process",
		action="store_true",
		help="Identify leftover artifacts within vocal output, may improve separation for some songs (default: %(default)s). Example: --enable_post_process",
	)
	vr_params.add_argument("--post_process_threshold", type=float, default=0.2, help="Threshold for post_process feature: 0.1-0.3 (default: %(default)s). Example: --post_process_threshold=0.1")

	audio_params = parser.add_argument_group("Audio Params")
	audio_params.add_argument(
		"--wav_bit_depth", choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"], default="FLOAT", help="Bit depth for wav output (default: %(default)s). Example: --wav_bit_depth=PCM_32"
	)
	audio_params.add_argument("--flac_bit_depth", choices=["PCM_16", "PCM_24"], default="PCM_24", help="Bit depth for flac output (default: %(default)s). Example: --flac_bit_depth=PCM_24")
	audio_params.add_argument("--mp3_bit_rate", choices=["96k", "128k", "192k", "256k", "320k"], default="320k", help="Bit rate for mp3 output (default: %(default)s). Example: --mp3_bit_rate=320k")

	args = parser.parse_args()
	vr_inference(args)
