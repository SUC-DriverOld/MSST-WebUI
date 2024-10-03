# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import os
import sys
parrent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parrent_dir)
import argparse
import time
import librosa
import logging
import warnings
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn

from tqdm import tqdm
from utils.utils import demix, get_model_from_config

warnings.filterwarnings("ignore", category = UserWarning)
log_format = "%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s - %(message)s"
date_format = "%H:%M:%S"
logging.basicConfig(level = logging.INFO, format = log_format, datefmt = date_format)
logger = logging.getLogger(__name__)

def run_folder(model, args, config, device):
    start_time = time.time()
    model.eval()
    all_mixtures_path = glob.glob(args.input_folder + '/*.*')
    logger.info('Total files found: {}'.format(len(all_mixtures_path)))
    
    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)
    extra_store_dir = args.extra_store_dir
    if not os.path.isdir(extra_store_dir):
        extra_store_dir = args.store_dir

    instruments = config.training.instruments.copy()
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]
    
    all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")
    for path in all_mixtures_path:
        all_mixtures_path.set_postfix({'track': os.path.basename(path)})
        try:
            mix, sr = librosa.load(path, sr = 44100, mono = False)
        except Exception as e:
            logger.warning('Cannot read track: {}'.format(path))
            logger.warning('Error message: {}'.format(str(e)))
            continue

        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)

        mix_orig = mix.copy()
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mono = mix.mean(0)
                mean = mono.mean()
                std = mono.std()
                mix = (mix - mean) / std

        if args.use_tta:
            track_proc_list = [mix.copy(), mix[::-1].copy(), -1. * mix.copy()]
        else:
            track_proc_list = [mix.copy()]

        full_result = []
        for mix in track_proc_list:
            waveforms = demix(config, model, mix, device, pbar=True, model_type=args.model_type)
            full_result.append(waveforms)

        waveforms = full_result[0]
        for i in range(1, len(full_result)):
            d = full_result[i]
            for el in d:
                if i == 2:
                    waveforms[el] += -1.0 * d[el]
                elif i == 1:
                    waveforms[el] += d[el][::-1].copy()
                else:
                    waveforms[el] += d[el]
        for el in waveforms:
            waveforms[el] = waveforms[el] / len(full_result)

        for instr in instruments:
            estimates = waveforms[instr].T
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = estimates * std + mean
            file_name, _ = os.path.splitext(os.path.basename(path))
            if args.instrumental_only and config.training.target_instrument is not None:
                pass
            else:
                save_separated_files(args, sr, file_name, instr, estimates, extra_store_dir)

        if (args.extract_instrumental and config.training.target_instrument is not None) or (args.instrumental_only and config.training.target_instrument is not None):
            if 'vocals' in instruments:
                extract_instrumental = 'instrumental'
            else:
                insts = config.training.instruments.copy()
                extract_instrumental = insts[1]
            waveforms[extract_instrumental] = mix_orig - waveforms[config.training.target_instrument]
            estimates = waveforms[extract_instrumental].T
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = estimates * std + mean
            file_name, _ = os.path.splitext(os.path.basename(path))
            save_separated_files(args, sr, file_name, extract_instrumental, estimates, extra_store_dir, isExtra=True)

    logger.info("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    logger.info('Results are saved to: {}'.format(args.store_dir))

def save_separated_files(args, sr, file_name, instr, estimates, extra_store_dir, isExtra=False):
    if isExtra:
        store_dir = extra_store_dir
    else:
        store_dir = args.store_dir
    if args.output_format.lower() == 'flac':
        output_file = os.path.join(store_dir, f"{file_name}_{instr}.flac")
        sf.write(output_file, estimates, sr, subtype='PCM_24')
    elif args.output_format.lower() == 'mp3':
        output_file = os.path.join(store_dir, f"{file_name}_{instr}.mp3")
        sf.write(output_file, estimates, sr, format='mp3')
    else:
        output_file = os.path.join(store_dir, f"{file_name}_{instr}.wav")
        sf.write(output_file, estimates, sr, subtype='FLOAT')

def proc_folder(args):
    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer, scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type = str, help = "path to config file")
    parser.add_argument("--start_check_point", type = str, default = '', help = "Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type = str, help = "folder with mixtures to process")
    parser.add_argument("--output_format", type = str, default = 'wav', help = "output format for separated files, one of wav, flac, mp3")
    parser.add_argument("--store_dir", default = "", type = str, help = "path to store results files")
    parser.add_argument("--device_ids", nargs = '+', type = int, default = 0, help = 'list of gpu ids')
    parser.add_argument("--extract_instrumental", action = 'store_true', help = "invert vocals to get instrumental if provided")
    parser.add_argument("--instrumental_only", action = 'store_true', help = "extract instrumental only")
    parser.add_argument("--extra_store_dir", default = "", type = str, help = "path to store extracted instrumental. If not provided, store_dir will be used")
    parser.add_argument("--force_cpu", action = 'store_true', help = "Force the use of CPU even if CUDA is available")
    parser.add_argument("--use_tta", action='store_true', help="Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality.")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        device = f'cuda:{args.device_ids[0]}' if type(args.device_ids) == list else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    logger.info(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point != '':
        logger.info('Start from checkpoint: {}'.format(args.start_check_point))
        if args.model_type in ['htdemucs', 'apollo']:
            state_dict = torch.load(args.start_check_point, map_location=device, weights_only=False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            state_dict = torch.load(args.start_check_point, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    logger.info("Instruments: {}".format(config.training.instruments))
    
    if type(args.device_ids) == list and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids = args.device_ids)
    model = model.to(device)

    run_folder(model, args, config, device)

if __name__ == "__main__":
    proc_folder(None)