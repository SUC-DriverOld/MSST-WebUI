import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import soundfile as sf
from utils.ensemble import ensemble_audios


def main(args):
	os.makedirs(args.output_dir, exist_ok=True)
	filename = f"ensemble_{args.type}_{len(args.files)}_songs.wav"
	if not args.weights:
		weights = [1] * len(args.files)
	else:
		weights = args.weights
	res, sr = ensemble_audios(args.files, args.type, weights)
	sf.write(os.path.join(args.output_dir, filename), res, sr, "FLOAT")
	print(f"Ensemble result saved at: {os.path.join(args.output_dir, filename)}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Ensemble from audio files", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
	parser.add_argument("--files", type=str, required=True, nargs="+", help="Path to all audio-files to ensemble")
	parser.add_argument(
		"--type", type=str, default="avg_wave", choices=["avg_wave", "median_wave", "min_wave", "max_wave", "avg_fft", "median_fft", "min_fft", "max_fft"], help="Type of ensemble to perform."
	)
	parser.add_argument("--weights", type=float, nargs="+", help="Weights to create ensemble. Number of weights must be equal to number of files")
	parser.add_argument("--output_dir", default="results", type=str, help="Path to wav file where ensemble result will be stored")
	args = parser.parse_args()

	if args.weights:
		assert len(args.files) == len(args.weights), "Number of weights must be equal to number of files!"
	main(args)
