import os
import librosa
import logging
import soundfile as sf
import torch
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment

from utils.utils import demix, get_model_from_config
from utils.logger import get_logger, set_log_level

vr_model_map = "data\\vr_model_map.json"
class MSSeparator:
    def __init__(
            self,
            model_type,
            config_path,
            model_path,
            device = 'auto',
            device_ids = [0],
            output_format = 'wav',
            use_tta = False,
            normalize = False,
            store_dirs = {},
            logger = get_logger(),
            debug = False
    ):

        if not model_type:
            raise ValueError('model_type is required')
        if not config_path:
            raise ValueError('config_path is required')
        if not model_path:
            raise ValueError('model_path is required')

        self.model_type = model_type
        self.config_path = config_path
        self.model_path = model_path
        self.output_format = output_format
        self.use_tta = use_tta
        self.normalize = normalize
        self.store_dirs = store_dirs
        self.logger = logger
        self.debug = debug

        if self.debug:
            set_log_level(logger, logging.DEBUG)

        self.device = "cpu"
        self.device_ids = device_ids

        if device not in ['cpu', 'cuda', 'mps']:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.device = f'cuda:{self.device_ids[0]}'
            elif torch.backends.mps.is_available():
                self.device = "mps"
        else:
            self.device = device

        torch.backends.cudnn.benchmark = True
        self.logger.info(f'Using device: {self.device}')

        self.model, self.config = self.load_model()

    def load_model(self):
        model, config = get_model_from_config(self.model_type, self.config_path)

        if self.model_type == 'htdemucs':
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
        else:
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)

        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        model = model.to(self.device)

        return model, config

    def process_folder(self, input_folder):
        if not os.path.isdir(input_folder):
            raise ValueError(f"Input folder '{input_folder}' does not exist.")

        all_mixtures_path = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
        self.logger.info('Total files found: {}'.format(len(all_mixtures_path)))

        if not self.debug:
            all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")

        for path in all_mixtures_path:
            if not self.debug:
                all_mixtures_path.set_postfix({'track': os.path.basename(path)})

            try:
                mix, sr = librosa.load(path, sr=44100, mono=False)
            except Exception as e:
                self.logger.warning(f'Cannot process track: {path}, error: {str(e)}')
                continue

            if len(mix.shape) == 1:
                mix = np.stack([mix, mix], axis=0)
            if len(mix.shape) > 2:
                self.logger.warning(f'Cannot process none stereo track: {path}, shape: {mix.shape}')
                continue

            self.logger.debug(f"Starting separation process for audio_file: {path}")
            results = self.separate(mix)
            self.logger.debug(f"Separation audio_file: {path} completed.")

            file_name, _ = os.path.splitext(os.path.basename(path))

            for instr in self.config.training.instruments:
                save_dir = self.store_dirs.get(instr, "")
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_audio(results[instr], sr, f"{file_name}_{instr}", save_dir)

    def separate(self, mix):
        instruments = self.config.training.instruments.copy()
        if self.config.training.target_instrument is not None:
            instruments = [self.config.training.target_instrument]

        mix_orig = mix.copy()
        if 'normalize' in self.config.inference and self.config.inference['normalize']:
            mono = mix.mean(0)
            mean = mono.mean()
            std = mono.std()
            mix = (mix - mean) / std

        if self.use_tta:
            track_proc_list = [mix.copy(), mix[::-1].copy(), -1. * mix.copy()]
        else:
            track_proc_list = [mix.copy()]

        full_result = []
        for mix in track_proc_list:
            waveforms = demix(self.config, self.model, mix, self.device, pbar=True, model_type=self.model_type)
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

        results = {}

        for instr in instruments:
            estimates = waveforms[instr].T

            if 'normalize' in self.config.inference and self.config.inference['normalize']:
                estimates = estimates * std + mean

            results[instr] = estimates

        if self.config.training.target_instrument is not None:
            target_instrument = self.config.training.target_instrument
            other_instruments = [instr for instr in self.config.training.instruments if instr != target_instrument]

            self.logger.debug(f"Extracting instrumental from {target_instrument}...")
            self.logger.debug(f"Other instruments: {other_instruments}")

            if other_instruments:
                extract_instrumental = other_instruments[0]
                waveforms[extract_instrumental] = mix_orig - waveforms[target_instrument]
                estimates = waveforms[extract_instrumental].T

                if 'normalize' in self.config.inference and self.config.inference['normalize']:
                    estimates = estimates * std + mean

                results[extract_instrumental] = estimates

        return results

    def save_audio(self, audio, sr, file_name, store_dir):
        if self.output_format.lower() == 'flac':
            file = os.path.join(store_dir, file_name + '.flac')
            sf.write(file, audio, sr, subtype='PCM_24')

        elif self.output_format.lower() == 'mp3':
            file = os.path.join(store_dir, file_name + '.mp3')

            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)

            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=sr,
                sample_width=audio.dtype.itemsize,
                channels=2
                )

            audio_segment.export(file, format='mp3', bitrate='320k')

        else:
            file = os.path.join(store_dir, file_name + '.wav')
            sf.write(file, audio, sr, subtype='FLOAT')