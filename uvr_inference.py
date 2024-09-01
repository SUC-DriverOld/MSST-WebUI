import sys
import os
import argparse
import logging
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from models.vocal_remover.separator import Separator

def inference(parser, args):
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s - %(message)s", datefmt="%H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)

    if not hasattr(args, "audio_file"):
        parser.print_help()
        sys.exit(1)

    logger.info(f"Separator beginning with input file: {args.audio_file}")

    separator = Separator(
        log_formatter=log_formatter,
        log_level=log_level,
        model_file_dir=args.model_file_dir,
        output_dir=args.output_dir,
        extra_output_dir=args.extra_output_dir,
        output_format=args.output_format,
        normalization_threshold=args.normalization,
        output_single_stem=args.single_stem,
        invert_using_spec=args.invert_spect,
        use_cpu=args.use_cpu,
        save_another_stem=args.save_another_stem,
        vr_params={
            "batch_size": args.vr_batch_size,
            "window_size": args.vr_window_size,
            "aggression": args.vr_aggression,
            "enable_tta": args.vr_enable_tta,
            "enable_post_process": args.vr_enable_post_process,
            "post_process_threshold": args.vr_post_process_threshold,
            "high_end_process": args.vr_high_end_process,
        },
    )
    separator.load_model(model_filename=args.model_filename)
    output_files = separator.separate(args.audio_file)
    logger.info(f"Separation complete! Output file(s): {' '.join(output_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate audio file into different stems.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))

    parser.add_argument("audio_file", nargs="?", help="The audio file path to separate, in any common format. You can input file path or file folder path", default=argparse.SUPPRESS)
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging, equivalent to --log_level=debug.")

    model_filename_help = "model to use for separation (default: %(default)s). Example: -m 2_HP-UVR.pth"
    output_format_help = "output format for separated files, any common format (default: %(default)s). Example: --output_format=MP3"
    output_dir_help = "directory to write output files (default: <current dir>). Example: --output_dir=/app/separated"
    model_file_dir_help = "model files directory (default: %(default)s). Example: --model_file_dir=/app/models"
    extra_output_dir_help = "extra output directory for saving another stem. If not provided, output_dir will be used. Example: --extra_output_dir=/app/extra_output"

    io_params = parser.add_argument_group("Separation I/O Params")
    io_params.add_argument("-m", "--model_filename", default="1_HP-UVR.pth", help=model_filename_help)
    io_params.add_argument("--output_format", default="FLAC", help=output_format_help)
    io_params.add_argument("--output_dir", default=None, help=output_dir_help)
    io_params.add_argument("--model_file_dir", default="pretrain/VR_Models", help=model_file_dir_help)
    io_params.add_argument("--extra_output_dir", default=None, help=extra_output_dir_help)

    invert_spect_help = "invert secondary stem using spectogram (default: %(default)s). Example: --invert_spect"
    normalization_help = "max peak amplitude to normalize input and output audio to (default: %(default)s). Example: --normalization=0.7"
    single_stem_help = "output only single stem, e.g. Instrumental, Vocals, Drums, Bass, Guitar, Piano, Other. Example: --single_stem=Instrumental"
    save_another_stem_help = "save another stem when using flow inference (default: %(default)s). Example: --save_another_stem"

    common_params = parser.add_argument_group("Common Separation Parameters")
    common_params.add_argument("--invert_spect", action="store_true", help=invert_spect_help)
    common_params.add_argument("--normalization", type=float, default=0.9, help=normalization_help)
    common_params.add_argument("--single_stem", default=None, help=single_stem_help)
    common_params.add_argument("--use_cpu", action="store_true", help="use CPU instead of GPU for inference")
    common_params.add_argument("--save_another_stem", action="store_true", help=save_another_stem_help)

    vr_batch_size_help = "number of batches to process at a time. higher = more RAM, slightly faster processing (default: %(default)s). Example: --vr_batch_size=16"
    vr_window_size_help = "balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: %(default)s). Example: --vr_window_size=320"
    vr_aggression_help = "intensity of primary stem extraction, -100 - 100. typically 5 for vocals & instrumentals (default: %(default)s). Example: --vr_aggression=2"
    vr_enable_tta_help = "enable Test-Time-Augmentation; slow but improves quality (default: %(default)s). Example: --vr_enable_tta"
    vr_high_end_process_help = "mirror the missing frequency range of the output (default: %(default)s). Example: --vr_high_end_process"
    vr_enable_post_process_help = "identify leftover artifacts within vocal output; may improve separation for some songs (default: %(default)s). Example: --vr_enable_post_process"
    vr_post_process_threshold_help = "threshold for post_process feature: 0.1-0.3 (default: %(default)s). Example: --vr_post_process_threshold=0.1"

    vr_params = parser.add_argument_group("VR Architecture Parameters")
    vr_params.add_argument("--vr_batch_size", type=int, default=4, help=vr_batch_size_help)
    vr_params.add_argument("--vr_window_size", type=int, default=512, help=vr_window_size_help)
    vr_params.add_argument("--vr_aggression", type=int, default=5, help=vr_aggression_help)
    vr_params.add_argument("--vr_enable_tta", action="store_true", help=vr_enable_tta_help)
    vr_params.add_argument("--vr_high_end_process", action="store_true", help=vr_high_end_process_help)
    vr_params.add_argument("--vr_enable_post_process", action="store_true", help=vr_enable_post_process_help)
    vr_params.add_argument("--vr_post_process_threshold", type=float, default=0.2, help=vr_post_process_threshold_help)

    args = parser.parse_args()
    inference(parser, args)