import argparse
import warnings
import logging
from time import time
from inference.msst_infer import MSSeparator
from utils.logger import get_logger
from utils.constant import MODEL_TYPE

def msst_inference(args):
    logger = get_logger(console_level=logging.INFO)

    if not args.debug:
        warnings.filterwarnings("ignore", category=UserWarning)

    if type(args.device_ids) == int:
        device_ids = [args.device_ids]

    start_time = time()

    separator = MSSeparator(
        model_type=args.model_type,
        config_path=args.config_path,
        model_path=args.model_path,
        device=args.device,
        device_ids=device_ids,
        output_format=args.output_format,
        use_tta=args.use_tta,
        store_dirs=args.output_folder,
        logger=logger,
        debug=args.debug
    )
    success_files = separator.process_folder(args.input_folder)
    separator.del_cache()
    logger.info(f"Successfully separated files: {success_files}, total time: {time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Music Source Separation Command Line Interface", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))

    parser.add_argument("-d", "--debug", action='store_true', help="Enable debug logging (default: %(default)s). Example: --debug")
    parser.add_argument("--device", default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help="Device to use for inference (default: %(default)s). Example: --device=cuda")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='List of gpu ids, only used when device is cuda (default: %(default)s). Example: --device_ids 0 1')

    io_params = parser.add_argument_group("Separation I/O Params")
    io_params.add_argument("-i", "--input_folder", type=str, help="Folder with mixtures to process. [required]")
    io_params.add_argument("-o", "--output_folder", default="results", help="Folder to store separated files. str for single folder, dict with instrument keys for multiple folders. Example: --output_folder=results or --output_folder=\"{'vocals': 'results/vocals', 'instrumental': 'results/instrumental'}\"")
    io_params.add_argument("--output_format", choices=['wav', 'flac', 'mp3'], default="wav", help="Output format for separated files (default: %(default)s). Example: --output_format=wav")

    model_params = parser.add_argument_group("Model Params")
    model_params.add_argument("--model_type", type=str, default='mdx23c', help=f"One of {MODEL_TYPE}. [required]")
    model_params.add_argument("--model_path", type=str, help="Path to model checkpoint. [required]")
    model_params.add_argument("--config_path", type=str, help="Path to config file. [required]")
    model_params.add_argument("--use_tta", action='store_true', help="Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality (default: %(default)s). Example: --use_tta")

    args = parser.parse_args()
    msst_inference(args)