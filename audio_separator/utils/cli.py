#!/usr/bin/env python
import torch
import logging
import json
import sys


def inference(parser, args):

    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s - %(message)s", datefmt="%H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level.upper())

    logger.setLevel(log_level)

    from audio_separator.separator import Separator

    if args.env_info:
        separator = Separator()
        sys.exit(0)

    if args.list_models:
        separator = Separator()
        print(json.dumps(separator.list_supported_model_files(), indent=4, sort_keys=True))
        sys.exit(0)

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
        sample_rate=args.sample_rate,
        use_cpu=args.use_cpu,
        save_another_stem=args.save_another_stem,
        mdx_params={
            "hop_length": args.mdx_hop_length,
            "segment_size": args.mdx_segment_size,
            "overlap": args.mdx_overlap,
            "batch_size": args.mdx_batch_size,
            "enable_denoise": args.mdx_enable_denoise,
        },
        vr_params={
            "batch_size": args.vr_batch_size,
            "window_size": args.vr_window_size,
            "aggression": args.vr_aggression,
            "enable_tta": args.vr_enable_tta,
            "enable_post_process": args.vr_enable_post_process,
            "post_process_threshold": args.vr_post_process_threshold,
            "high_end_process": args.vr_high_end_process,
        },
        demucs_params={"segment_size": args.demucs_segment_size, "shifts": args.demucs_shifts, "overlap": args.demucs_overlap, "segments_enabled": args.demucs_segments_enabled},
        mdxc_params={
            "segment_size": args.mdxc_segment_size,
            "batch_size": args.mdxc_batch_size,
            "overlap": args.mdxc_overlap,
            "override_model_segment_size": args.mdxc_override_model_segment_size,
            "pitch_shift": args.mdxc_pitch_shift,
        },
    )
    torch.cuda.empty_cache()

    separator.load_model(model_filename=args.model_filename)

    output_files = separator.separate(args.audio_file)

    logger.info(f"Separation complete! Output file(s): {' '.join(output_files)}")
