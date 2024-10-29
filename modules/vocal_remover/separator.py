""" This file contains the Separator class, to facilitate the separation of stems from audio. """
import gc
import os
import platform
import subprocess
import time
import logging
import json
import torch
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
from modules.vocal_remover.vr_separator import VRSeparator
from utils.logger import get_logger, set_log_level
from utils.constant import TEMP_PATH, VR_MODEL, UNOFFICIAL_MODEL

class Separator:
    def __init__(
        self,
        logger=get_logger(),
        debug=False,
        model_file="pretrain/VR_Models/1_HP-UVR.pth",
        output_dir="results",
        output_format="wav",
        invert_using_spec=False,
        use_cpu=False,
        vr_params={"batch_size": 2, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
    ):
        if debug:
            set_log_level(logger, logging.DEBUG)
        else:
            set_log_level(logger, logging.INFO)

        self.logger = logger
        self.debug = debug
        self.model_file = model_file
        self.output_dir = output_dir
        self.output_format = output_format
        self.invert_using_spec = invert_using_spec

        if self.invert_using_spec:
            self.logger.debug(f"Secondary step will be inverted using spectogram rather than waveform. This may improve quality but is slightly slower.")

        # These are parameters which users may want to configure so we expose them to the top-level Separator class,
        # even though they are specific to a single model architecture
        self.vr_params_params = vr_params
        self.sample_rate = 44100
        self.use_cpu = use_cpu
        self.torch_device = None
        self.torch_device_cpu = None
        self.torch_device_mps = None
        self.model_instance = None

        self.setup_accelerated_inferencing_device()
        self.load_model(self.model_file)

        if type(self.output_dir) == str:
            self.output_dir = {
                self.model_instance.primary_stem_name: self.output_dir,
                self.model_instance.secondary_stem_name: self.output_dir,
                }

        for key in list(self.output_dir.keys()):
            stem_name = [self.model_instance.primary_stem_name, self.model_instance.secondary_stem_name]
            if key not in stem_name and key.lower() not in stem_name:
                self.output_dir.pop(key)
                self.logger.warning(f"Invalid instrument key: {key}, removing from output_dir")
                self.logger.warning(f"Valid instrument keys: {stem_name}")

    def setup_accelerated_inferencing_device(self):
        """
        This method sets up the PyTorch inferencing device, using GPU hardware acceleration if available.
        """
        self.log_system_info()
        self.check_ffmpeg_installed()
        if self.use_cpu:
            self.logger.info("CPU inference requested, ignoring GPU availability")
            self.torch_device_cpu = torch.device("cpu")
            self.torch_device = self.torch_device_cpu
        else:
            self.setup_torch_device()

    def log_system_info(self):
        """
        This method logs the system information, including the operating system, CPU archutecture and Python version
        """
        os_name = platform.system()
        os_version = platform.version()
        self.logger.debug(f"Operating System: {os_name} {os_version}")

        python_version = platform.python_version()
        self.logger.debug(f"Python Version: {python_version}")

        pytorch_version = torch.__version__
        self.logger.debug(f"PyTorch Version: {pytorch_version}")

    def check_ffmpeg_installed(self):
        """
        This method checks if ffmpeg is installed and logs its version.
        """
        try:
            ffmpeg_version_output = subprocess.check_output(["ffmpeg", "-version"], text=True)
            first_line = ffmpeg_version_output.splitlines()[0]
            self.logger.debug(f"FFmpeg installed: {first_line}")
        except FileNotFoundError:
            self.logger.error("FFmpeg is not installed. Please install FFmpeg to use this package.")

    def setup_torch_device(self):
        """
        This method sets up the PyTorch inferencing device, using GPU hardware acceleration if available.
        """
        hardware_acceleration_enabled = False
        self.torch_device_cpu = torch.device("cpu")

        if torch.cuda.is_available():
            self.configure_cuda()
            hardware_acceleration_enabled = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.configure_mps()
            hardware_acceleration_enabled = True

        if not hardware_acceleration_enabled:
            self.logger.info("No hardware acceleration could be configured, running in CPU mode")
            self.torch_device = self.torch_device_cpu

    def configure_cuda(self):
        self.logger.info("CUDA is available in Torch, setting Torch device to CUDA")
        self.torch_device = torch.device("cuda")

    def configure_mps(self):
        self.logger.info("Apple Silicon MPS/CoreML is available in Torch, setting Torch device to MPS")
        self.torch_device_mps = torch.device("mps")
        self.torch_device = self.torch_device_mps

    def load_model_data(self, model_name):
        vr_model_data_object = json.load(open(VR_MODEL, encoding="utf-8"))
        if model_name in vr_model_data_object.keys():
            model_data = vr_model_data_object[model_name]
            self.logger.debug(f"Model data loaded from UVR JSON: {model_data}")
            return model_data
        else:
            unofficial_model_data = json.load(open(os.path.join(UNOFFICIAL_MODEL, "unofficial_vr_model.json"), encoding="utf-8"))
            model_data = unofficial_model_data[model_name]
            self.logger.debug(f"Model data loaded from unofficial UVR JSON: {model_data}")
            return model_data

    def load_model(self, model_path):
        model_filename = os.path.basename(model_path)
        self.logger.debug(f"Loading model {model_filename}...")

        load_model_start_time = time.time()
        model_name = model_filename.split(".")[0]
        model_data = self.load_model_data(model_filename)

        common_params = {
            "logger": self.logger,
            "debug": self.debug,
            "torch_device": self.torch_device,
            "torch_device_cpu": self.torch_device_cpu,
            "torch_device_mps": self.torch_device_mps,
            "model_name": model_name,
            "model_path": model_path,
            "model_data": model_data,
            "output_format": self.output_format,
            "output_dir": self.output_dir,
            "invert_using_spec": self.invert_using_spec,
            "sample_rate": self.sample_rate,
        }

        self.model_instance = VRSeparator(common_config=common_params, arch_config=self.vr_params_params)
        self.model_instance.load_model()
        self.logger.debug(f'Loading model completed, duration: {time.time() - load_model_start_time:.2f} seconds')

    def process_folder(self, input_folder):
        if not os.path.isdir(input_folder):
            raise ValueError(f"Input folder '{input_folder}' does not exist.")

        all_audio_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
        self.logger.info(f"Input_folder: {input_folder}, Total files found: {len(all_audio_files)}")

        if not self.debug:
            all_audio_files = tqdm(all_audio_files, desc="Total progress")

        success_files = []
        for file_path in all_audio_files:
            if not self.debug:
                all_audio_files.set_postfix({"track": file_path})

            try:
                mix, sr = librosa.load(file_path, sr=44100, mono=False)
            except Exception as e:
                self.logger.warning(f'Cannot process track: {file_path}, error: {str(e)}')
                continue

            self.logger.debug(f"Starting separation process for audio_file: {file_path}")
            results = self.separate(file_path)
            self.logger.debug(f"Separation audio_file: {file_path} completed. Starting to save results.")

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            for stem in results.keys():
                store_dir = self.output_dir.get(stem, "")
                if store_dir and type(store_dir) == str:
                    os.makedirs(store_dir, exist_ok=True)
                    self.save_audio(results[stem], sr, f"{base_name}_{stem}", store_dir)
                    self.logger.debug(f"Saved {stem} for {base_name}_{stem}.{self.output_format} in {store_dir}")
                elif store_dir and type(store_dir) == list:
                    for dir in store_dir:
                        os.makedirs(dir, exist_ok=True)
                        self.save_audio(results[stem], sr, f"{base_name}_{stem}", dir)
                        self.logger.debug(f"Saved {stem} for {base_name}_{stem}.{self.output_format} in {dir}")

            success_files.append(os.path.basename(file_path))
        return success_files

    def separate(self, mix):
        is_numpy = isinstance(mix, np.ndarray)
        if is_numpy:
            tempdir = os.path.join(TEMP_PATH, "tmp_audio")
            os.makedirs(tempdir, exist_ok=True)
            sf.write(os.path.join(tempdir, "tmp_audio.wav"), mix.T, 44100, subtype="FLOAT")

            mix = os.path.join(tempdir, "tmp_audio.wav")
            self.logger.debug(f"Temporary audio file created: {mix}")

        results = self.model_instance.separate(mix)
        self.model_instance.clear_file_specific_paths()

        if is_numpy and os.path.exists(mix):
            os.remove(mix)

        self.logger.debug("Separation process completed.")

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

    def del_cache(self):
        self.logger.debug("Running garbage collection...")
        gc.collect()
        if self.torch_device == torch.device("mps"):
            self.logger.debug("Clearing MPS cache...")
            torch.mps.empty_cache()
        if self.torch_device == torch.device("cuda"):
            self.logger.debug("Clearing CUDA cache...")
            torch.cuda.empty_cache()