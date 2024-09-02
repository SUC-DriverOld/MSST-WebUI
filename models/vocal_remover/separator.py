""" This file contains the Separator class, to facilitate the separation of stems from audio. """

import os
import platform
import subprocess
import time
import logging
import warnings
import json
import torch
from models.vocal_remover.vr_separator import VRSeparator

VR_MODEL_MAP = "data/vr_model_map.json"
UNOFFICIAL_MODEL_MAP = "config_unofficial/unofficial_vr_model.json"

class Separator:
    def __init__(
        self,
        log_level=logging.INFO,
        log_formatter=None,
        model_file_dir="pretrain/VR_Models",
        output_dir=None,
        extra_output_dir=None,
        output_format="wav",
        normalization_threshold=0.9,
        output_single_stem=None,
        invert_using_spec=False,
        sample_rate=44100,
        use_cpu=False,
        save_another_stem=False,
        vr_params={"batch_size": 16, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.log_formatter = log_formatter

        self.log_handler = logging.StreamHandler()

        if self.log_formatter is None:
            self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        self.log_handler.setFormatter(self.log_formatter)

        if not self.logger.hasHandlers():
            self.logger.addHandler(self.log_handler)

        # Filter out noisy warnings from PyTorch for users who don't care about them
        if log_level > logging.DEBUG:
            warnings.filterwarnings("ignore")

        self.model_file_dir = model_file_dir
        
        self.save_another_stem = save_another_stem
        if output_single_stem is None and save_another_stem:
            self.logger.warning("The save_another_stem option is only applicable when output_single_stem is set. Ignoring save_another_stem.")
            self.save_another_stem = False

        if output_dir is None:
            output_dir = os.getcwd()
            self.logger.info("Output directory not specified. Using current working directory.")

        self.output_dir = output_dir
        self.logger.info(f"Separator instantiating with output_dir: {output_dir}, output_format: {output_format}")

        if extra_output_dir is None:
            extra_output_dir = output_dir
            if self.save_another_stem:
                self.logger.info(f"Extra output directory not specified. Using output directory: {extra_output_dir} as extra output directory.")
        
        self.extra_output_dir = extra_output_dir
        if self.save_another_stem:
            self.logger.info(f"Separator instantiating with extra_output_dir: {extra_output_dir}")

        # Create the model directory if it does not exist
        os.makedirs(self.model_file_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.output_format = output_format

        if self.output_format is None:
            self.output_format = "wav"

        self.normalization_threshold = normalization_threshold
        if normalization_threshold <= 0 or normalization_threshold > 1:
            raise ValueError("The normalization_threshold must be greater than 0 and less than or equal to 1.")

        self.output_single_stem = output_single_stem
        if output_single_stem is not None:
            self.logger.debug(f"Single stem output requested, so only one output file ({output_single_stem}) will be written")

        self.invert_using_spec = invert_using_spec
        if self.invert_using_spec:
            self.logger.debug(f"Secondary step will be inverted using spectogram rather than waveform. This may improve quality but is slightly slower.")

        # These are parameters which users may want to configure so we expose them to the top-level Separator class,
        # even though they are specific to a single model architecture
        self.vr_params_params = vr_params
        self.sample_rate = sample_rate
        self.use_cpu = use_cpu
        self.torch_device = None
        self.torch_device_cpu = None
        self.torch_device_mps = None
        self.model_instance = None

        self.setup_accelerated_inferencing_device()

    def setup_accelerated_inferencing_device(self):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
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

        system_info = platform.uname()
        self.logger.debug(f"System: {system_info.system} Node: {system_info.node} Release: {system_info.release} Machine: {system_info.machine} Proc: {system_info.processor}")

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
            # Raise an exception if this is being run by a user, as ffmpeg is required for pydub to write audio
            # but if we're just running unit tests in CI, no reason to throw
            if "PYTEST_CURRENT_TEST" not in os.environ:
                raise

    def setup_torch_device(self):
        """
        This method sets up the PyTorch and/or ONNX Runtime inferencing device, using GPU hardware acceleration if available.
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
        vr_model_data_object = json.load(open(VR_MODEL_MAP, encoding="utf-8"))
        if model_name in vr_model_data_object.keys():
            model_data = vr_model_data_object[model_name]
            self.logger.debug(f"Model data loaded from UVR JSON: {model_data}")
            return model_data
        else:
            unofficial_model_data = json.load(open(UNOFFICIAL_MODEL_MAP, encoding="utf-8"))
            model_data = unofficial_model_data[model_name]
            self.logger.debug(f"Model data loaded from unofficial UVR JSON: {model_data}")
            return model_data

    def load_model(self, model_filename="1_HP-UVR.pth"):
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
        }

        self.logger.debug(f"Instantiating vr_separator class")
        self.model_instance = VRSeparator(common_config=common_params, arch_config=self.vr_params_params)
        self.logger.debug("Loading model completed.")
        self.logger.info(f'Load model duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - load_model_start_time)))}')

    def separate_onefile(self, file_path):
        self.logger.info(f"Starting separation process for audio_file_path: {file_path}")
        separate_start_time = time.perf_counter()
        self.logger.debug(f"Normalization threshold set to {self.normalization_threshold}, waveform will be lowered to this max amplitude to avoid clipping.")

        file_output_files = self.model_instance.separate(file_path)
        self.model_instance.clear_gpu_cache()
        self.model_instance.clear_file_specific_paths()

        self.logger.debug("Separation process completed.")
        self.logger.info(f'Separation duration: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - separate_start_time)))}')

        return file_output_files

    def separate(self, folder_path):
        if os.path.isfile(folder_path):
            return self.separate_onefile(folder_path)
        output_files = []

        if not os.path.isdir(folder_path):
            self.logger.error(f"The provided folder path does not exist: {folder_path}")
            return output_files

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if not filename.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.m4a', '.ogg', '.wma')): 
                self.logger.warning(f"Skipping not supported audio file: {filename}")
                continue

            file_output_files = self.separate_onefile(file_path)
            output_files.extend(file_output_files)

        return output_files
