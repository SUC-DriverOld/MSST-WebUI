import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
from tools.SOME.infer import infer


def main(args):
	os.makedirs(args.output_dir, exist_ok=True)
	midi_path = infer(args.model, args.config, args.input_audio, args.output_dir, args.tempo)
	print(f"Output MIDI file: {midi_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="SOME Command Line Interface", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
	parser.add_argument("-m", "--model", type=str, default="tools/SOME_weights/model_steps_64000_simplified.ckpt", help="Path to the model checkpoint (*.ckpt)")
	parser.add_argument("-c", "--config", type=str, default="configs_backup/config_some.yaml", help="Path to the config file (*.yaml)")
	parser.add_argument("-i", "--input_audio", type=str, help="Path to the input audio file", required=True)
	parser.add_argument("-o", "--output_dir", type=str, default="results", help="Path to the output folder")
	parser.add_argument("-t", "--tempo", type=float, default=120, help="Specify tempo in the output MIDI")
	args = parser.parse_args()

	if not args.input_audio:
		raise ValueError("Please specify the input audio file")
	if not os.path.exists(args.model):
		raise ValueError("Model checkpoint not found")
	if not os.path.exists(args.config):
		raise ValueError("Config file not found")

	main(args)
