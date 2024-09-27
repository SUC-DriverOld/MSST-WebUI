import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

import librosa
import logging
import soundfile as sf
import torch
import numpy as np
import time
import math
import audioread
from tqdm import tqdm
import json
from utils import demix, get_model_from_config
from models.vocal_remover.separator import Separator
from models.vocal_remover.vr_separator import VRSeparator
from models.vocal_remover.uvr_lib_v5 import spec_utils
from models.vocal_remover.uvr_lib_v5.vr_network import nets
from models.vocal_remover.uvr_lib_v5.vr_network import nets_new

logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s - %(message)s",
                                  datefmt = "%H:%M:%S")
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)
log_level = logging.INFO
logger.setLevel(log_level)


def load_audio(audio_file: str):
    try:
        audio, sr = librosa.load(audio_file, sr = 44100, mono = False)
    except Exception as e:
        raise AssertionError(f'Cannot read track: {audio_file}, error message: {str(e)}')
    return audio, sr


def save_audio(audio, sr: int, audio_type: str, file_name: str, store_dir: str) -> None:
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
    if audio_type.lower() == 'flac':
        file = os.path.join(store_dir, file_name + '.flac')
        sf.write(file, audio, sr, subtype = 'PCM_24')
    elif audio_type.lower() == 'mp3':
        file = os.path.join(store_dir, file_name + '.mp3')
        sf.write(file, audio, sr, format = 'mp3')
    else:
        file = os.path.join(store_dir, file_name + '.wav')
        sf.write(file, audio, sr, subtype = 'FLOAT')


class ComfyMSST:
    def __init__(
            self,
            model_type,
            config_path,
            model_path,
            device = None,
            output_format = 'wav',
            use_tta = False,
            normalize = False,
            store_dirs = None
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
        self.store_dirs = store_dirs if store_dirs else {}

        if device not in ['cpu', 'cuda', 'mps']:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
        else:
            self.device = device

        torch.backends.cudnn.benchmark = True
        logger.info(f'Using device: {self.device}')

        self.model, self.config = self.load_model()

    def load_model(self):
        model, config = get_model_from_config(self.model_type, self.config_path)

        if self.model_type == 'htdemucs':
            state_dict = torch.load(self.model_path, map_location = self.device, weights_only = False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
        else:
            state_dict = torch.load(self.model_path, map_location = self.device, weights_only = True)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        return model, config

    def process_folder(self, input_folder):
        if not os.path.isdir(input_folder):
            raise ValueError(f"Input folder '{input_folder}' does not exist.")

        all_mixtures_path = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
        logger.info('Total files found: {}'.format(len(all_mixtures_path)))

        all_mixtures_path = tqdm(all_mixtures_path, desc = "Total progress")
        for path in all_mixtures_path:
            all_mixtures_path.set_postfix({'track': os.path.basename(path)})
            try:
                results = self.separate(path)
                file_name, _ = os.path.splitext(os.path.basename(path))
                for instr in self.config.training.instruments:  # 使用模型的乐器信息
                    save_dirs = self.store_dirs.get(instr)
                    if save_dirs:  # 只有在 save_dir 不为空或 None 时才保存
                        for save_dir in save_dirs:
                            save_audio(results[instr], 44100, self.output_format, f"{file_name}_{instr}", save_dir)
            except Exception as e:
                logger.warning('Cannot process track: {}'.format(path))
                logger.warning('Error message: {}'.format(str(e)))

    def separate(self, input_file):
        instruments = self.config.training.instruments.copy()
        if self.config.training.target_instrument is not None:
            instruments = [self.config.training.target_instrument]
        mix, sr = load_audio(input_file)

        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis = 0)

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
        logger.info("Start demixing...")
        for mix in track_proc_list:
            waveforms = demix(self.config, self.model, mix, self.device, pbar = True, model_type = self.model_type)
            logger.info("Demixing completed.")
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
        logger.info(f"instruments: {instruments}")

        for instr in instruments:
            estimates = waveforms[instr].T
            if 'normalize' in self.config.inference and self.config.inference['normalize']:
                estimates = estimates * std + mean
            results[instr] = estimates

        # 处理 target_instrument
        if self.config.training.target_instrument is not None:
            target_instrument = self.config.training.target_instrument
            other_instruments = [instr for instr in self.config.training.instruments if instr != target_instrument]
            logger.info(f"Extracting instrumental from {target_instrument}...")
            logger.info(f"Other instruments: {other_instruments}")
            if other_instruments:
                extract_instrumental = other_instruments[0]
                waveforms[extract_instrumental] = mix_orig - waveforms[target_instrument]
                estimates = waveforms[extract_instrumental].T
                if 'normalize' in self.config.inference and self.config.inference['normalize']:
                    estimates = estimates * std + mean
                results[extract_instrumental] = estimates

        return results


class ComfyVR(Separator):
    def __init__(
            self,
            log_level = logging.DEBUG,
            log_formatter = log_formatter,
            model_file = "pretrain/VR_Models/1_HP-UVR.pth",
            output_format = "wav",
            normalization_threshold = 0.9,
            output_single_stem = None,
            invert_using_spec = False,
            use_cpu = False,
            vr_params = {"batch_size": 2, "window_size": 512, "aggression": 5, "enable_tta": False,
                         "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
            store_dirs = {}
    ):
        """Initialize the ComfyVR class
        Args:
            model_file_dir (str): Path to the model files directory
            output_format (str, optional): One of 'wav', 'flac', 'mp3'. Defaults to 'wav'.
            normalization_threshold (float, optional): Defaults to 0.9.
            output_single_stem (str, optional): Defaults to None.
            invert_using_spec (bool, optional): Defaults to False.
            sample_rate (int, optional): Defaults to 44100.
            use_cpu (bool, optional): Defaults to False.
            vr_params (dict, optional): Defaults to {"batch_size": 16, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False}.
            store_dirs (dict, optional): Dictionary for specifying output paths for each stem.
        Returns:
            None
        """
        self.store_dirs = store_dirs
        model_file_dir, model_name = os.path.split(model_file)
        super().__init__(
            log_level = log_level,
            log_formatter = log_formatter,
            model_file_dir = model_file_dir,
            output_format = output_format,
            normalization_threshold = normalization_threshold,
            output_single_stem = output_single_stem,
            invert_using_spec = invert_using_spec,
            use_cpu = use_cpu,
            vr_params = vr_params,
        )

        self.load_model(model_name)

    def load_model(self, model_filename):
        self.logger.info(f"Loading model {model_filename}...")

        load_model_start_time = time.perf_counter()
        model_path = os.path.join(self.model_file_dir, f"{model_filename}")
        model_name = model_filename.split(".")[0]
        model_data = self.load_model_data(model_filename)

        common_params = {
            "logger": self.logger,
            "log_level": self.log_level,
            "torch_device": self.torch_device,
            "torch_device_cpu": self.torch_device_cpu,
            "torch_device_mps": self.torch_device_mps,
            "model_name": model_name,
            "model_path": model_path,
            "model_data": model_data,
            "output_format": self.output_format,
            "output_dir": self.output_dir,
            "extra_output_dir": self.extra_output_dir,
            "normalization_threshold": self.normalization_threshold,
            "output_single_stem": self.output_single_stem,
            "invert_using_spec": self.invert_using_spec,
            "sample_rate": self.sample_rate,
            "save_another_stem": self.save_another_stem,
            "store_dirs": self.store_dirs,
        }

        self.logger.debug(f"Instantiating vr_separator class")
        self.model_instance = ComfyVRSeparator(common_config = common_params, arch_config = self.vr_params_params)
        self.logger.debug("Loading model completed.")
        self.logger.debug(
            f'Load model duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - load_model_start_time)))}')

    def separate(self, folder_path):
        """Process all files in the folder
        Args:
            folder_path (str): The path of the folder containing the audio files to be processed
        """
        self.logger.debug(f"Starting separation process for folder: {folder_path}")

        # Only process the audio files in the specified folder (no recursion)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.wav', '.flac', '.mp3')):
                self.logger.info(f"Processing file: {file_path}")
                file_output_files = self.model_instance.VRseparate(file_path)
                self.model_instance.clear_gpu_cache()
                self.model_instance.clear_file_specific_paths()

                self.logger.info(f"Save VR results for {file_path}")

                for inst in file_output_files.keys():
                    self.model_instance.process_stem(inst, file_output_files[inst], file_output_files[inst])

        self.logger.debug(f"Separation process for folder: {folder_path} completed.")


class ComfyVRSeparator(VRSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(common_config, arch_config)
        self.common_config = common_config

    def VRseparate(self, audio_file_path):
        self.primary_source = None
        self.secondary_source = None

        with open("data\\vr_model_map.json", 'r') as f:
            model_map = json.load(f)

        self.primary_stem_name = model_map[self.model_name + '.pth']["primary_stem"]
        self.secondary_stem_name = model_map[self.model_name + '.pth']["secondary_stem"]

        if os.path.isfile(audio_file_path):
            self.audio_file_path = audio_file_path
            self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        else:
            self.audio_file_path = audio_file_path
            self.audio_file_base = "numpy_array"

        nn_arch_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
        vr_5_1_models = [56817, 218409]
        model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
        nn_arch_size = min(nn_arch_sizes, key = lambda x: abs(x - model_size))

        if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
            self.logger.debug("Using CascadedNet for VR 5.1 model...")
            self.model_run = nets_new.CascadedNet(self.model_params.param["bins"] * 2, nn_arch_size,
                                                  nout = self.model_capacity[0], nout_lstm = self.model_capacity[1])
            self.is_vr_51_model = True
        else:
            self.logger.debug("Determining model capacity...")
            self.model_run = nets.determine_model_capacity(self.model_params.param["bins"] * 2, nn_arch_size)

        self.model_run.load_state_dict(torch.load(self.model_path, map_location = self.torch_device_cpu))
        self.model_run.to(self.torch_device)
        self.logger.debug("Model loaded and moved to device.")

        y_spec, v_spec = self.inference_vr(self.loading_mix(), self.torch_device, self.aggressiveness)
        self.logger.debug("Inference completed.")

        y_spec = np.nan_to_num(y_spec, nan = 0.0, posinf = 0.0, neginf = 0.0)
        v_spec = np.nan_to_num(v_spec, nan = 0.0, posinf = 0.0, neginf = 0.0)

        self.logger.debug("Sanitization completed. Replaced NaN and infinite values in y_spec and v_spec.")
        self.logger.debug(f"Inference VR completed. y_spec shape: {y_spec.shape}, v_spec shape: {v_spec.shape}")
        self.logger.debug(
            f"y_spec stats - min: {np.min(y_spec)}, max: {np.max(y_spec)}, isnan: {np.isnan(y_spec).any()}, isinf: {np.isinf(y_spec).any()}")
        self.logger.debug(
            f"v_spec stats - min: {np.min(v_spec)}, max: {np.max(v_spec)}, isnan: {np.isnan(v_spec).any()}, isinf: {np.isinf(v_spec).any()}")

        output_files = {}
        self.logger.debug("Processing output files...")

        output_files[self.primary_stem_name] = self.process_stem(self.primary_stem_name, self.primary_source, y_spec)
        output_files[self.secondary_stem_name] = self.process_stem(self.secondary_stem_name, self.secondary_source,
                                                                   v_spec)

        return output_files

    def process_stem(self, stem_name, stem_source, spec):
        self.logger.debug(f"Processing {stem_name} stem")

        if not isinstance(stem_source, np.ndarray):
            self.logger.debug(f"Preparing to convert spectrogram to waveform. Spec shape: {spec.shape}")
            stem_source = self.spec_to_wav(spec).T
            self.logger.debug(f"Converting {stem_name} spectrogram to waveform.")
            if self.model_samplerate != 44100:
                stem_source = librosa.resample(stem_source.T, orig_sr = self.model_samplerate, target_sr = 44100).T
                self.logger.debug(f"Resampling {stem_name} to 44100Hz.")

        if self.common_config["store_dirs"].get(stem_name):
            store_dir = self.common_config["store_dirs"][stem_name]
            if store_dir:
                if self.audio_file_path:
                    file_name = f"{self.audio_file_base}_{stem_name}"
                    save_audio(stem_source, 44100, self.common_config["output_format"], file_name, store_dir)

        return stem_source

    def loading_mix(self):
        X_wave, X_spec_s = {}, {}
        bands_n = len(self.model_params.param["band"])
        audio_file = spec_utils.write_array_to_mem(self.audio_file_path, subtype = self.wav_subtype)
        is_mp3 = audio_file.endswith(".mp3") if isinstance(audio_file, str) else False
        self.logger.debug(f"loading_mix iterating through {bands_n} bands")

        if self.log_level == logging.DEBUG:
            process_bands_n = tqdm(range(bands_n, 0, -1))
        else:
            process_bands_n = tqdm(range(bands_n, 0, -1), leave = False)

        for d in process_bands_n:
            bp = self.model_params.param["band"][d]
            wav_resolution = bp["res_type"]
            if self.torch_device_mps is not None:
                wav_resolution = "polyphase"

            if d == bands_n:
                if type(audio_file) == np.ndarray:
                    X_wave[d] = audio_file
                else:
                    X_wave[d], _ = librosa.load(audio_file, sr = bp["sr"], mono = False, dtype = np.float32,
                                                res_type = wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.model_params,
                                                             band = d, is_v51_model = self.is_vr_51_model)

                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp["sr"])
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else:
                X_wave[d] = librosa.resample(X_wave[d + 1], orig_sr = self.model_params.param["band"][d + 1]["sr"],
                                             target_sr = bp["sr"], res_type = wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp["hl"], bp["n_fft"], self.model_params,
                                                             band = d, is_v51_model = self.is_vr_51_model)

            if d == bands_n and self.high_end_process:
                self.input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                        self.model_params.param["pre_filter_stop"] - self.model_params.param["pre_filter_start"])
                self.input_high_end = X_spec_s[d][:, bp["n_fft"] // 2 - self.input_high_end_h: bp["n_fft"] // 2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.model_params, is_v51_model = self.is_vr_51_model)
        del X_wave, X_spec_s, audio_file
        return X_spec


def rerun_mp3(audio_file, sample_rate = 44100):
    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)
    return librosa.load(audio_file, duration = track_length, mono = False, sr = sample_rate)[0]


if __name__ == "__main__":
    # vr分离示例
    # vr_separate = ComfyVR(
    #     model_file = 'pretrain/VR_Models/1_HP-UVR.pth',
    #     output_format = "flac",
    #     store_dirs = {
    #         'Vocals': 'output/vocals',
    #         'Instrumental': ''
    #     }
    #     # 省略部分参数
    # )
    #
    # vr_separate.separate("input")

    # msst分离示例
    # 留空表示不输出
    store_dirs = {
        'vocals': ['output/vocals'],
        'instrumental': ['output/instrumental', 'output/instrumental2'],
    }

    comfy_msst = ComfyMSST(
        model_type = "htdemucs",
        config_path = r"configs/multi_stem_models/config_musdb18_htdemucs.yaml",
        model_path = r"D:\projects\MSST-WebUI\pretrain\multi_stem_models\HTDemucs4.th",
        device = "cpu",
        output_format = 'wav',
        store_dirs = store_dirs,
        use_tta = False,
        normalize = False
    )
    comfy_msst.process_folder("input")
