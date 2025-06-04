import gc
import os
import librosa
import logging
import soundfile as sf
import torch
import numpy as np
import platform
import subprocess
from time import time
from tqdm import tqdm
from pydub import AudioSegment

from utils.utils import demix, get_model_from_config
from utils.logger import get_logger, set_log_level


class MSSeparator:
	def __init__(
		self,
		model_type,
		config_path,
		model_path,
		device="auto",
		device_ids=[0],
		output_format="wav",
		use_tta=False,
		store_dirs="results",  # str for single folder, dict with instrument keys for multiple folders
		audio_params={"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"},
		logger=get_logger(),
		debug=False,
		inference_params={"batch_size": None, "num_overlap": None, "chunk_size": None, "normalize": None},
		callback=None,
	):
		self.logger = logger

		if not model_type:
			raise ValueError("model_type is required")
		if not config_path:
			config_path = model_path.replace("pretrain", "configs") + ".yaml"
			logger.info(f"config_path is not provided, using default config_path: {config_path}")
		if not model_path:
			raise ValueError("model_path is required")

		self.model_type = model_type
		self.config_path = config_path
		self.model_path = model_path
		self.output_format = output_format
		self.use_tta = use_tta
		self.store_dirs = store_dirs
		self.audio_params = audio_params
		self.debug = debug
		self.callback = callback

		if self.debug:
			set_log_level(logger, logging.DEBUG)
		else:
			set_log_level(logger, logging.INFO)

		self.inference_params = inference_params

		self.log_system_info()
		self.check_ffmpeg_installed()

		self.device = "cpu"
		self.device_ids = device_ids

		if device not in ["cpu", "cuda", "mps"]:
			if torch.cuda.is_available():
				self.device = "cuda"
				self.device = f"cuda:{self.device_ids[0]}"
				self.logger.debug("CUDA is available in Torch, setting Torch device to CUDA")
			elif torch.backends.mps.is_available():
				self.device = "mps"
				self.logger.debug("Apple Silicon MPS/CoreML is available in Torch, setting Torch device to MPS")
		else:
			self.device = device

		if self.device == "cpu":
			self.logger.warning("No hardware acceleration could be configured, running in CPU mode")

		torch.backends.cudnn.benchmark = True
		self.logger.info(f"Using device: {self.device}, device_ids: {self.device_ids}")

		self.model, self.config = self.load_model()

		if type(self.store_dirs) == str:
			self.store_dirs = {k: self.store_dirs for k in self.config.training.instruments}

		for key in list(self.store_dirs.keys()):
			if key not in self.config.training.instruments and key.lower() not in self.config.training.instruments:
				self.store_dirs.pop(key)
				self.logger.warning(f"Invalid instrument key: {key}, removing from store_dirs")
				self.logger.warning(f"Valid instrument keys: {self.config.training.instruments}")

	def log_system_info(self):
		os_name = platform.system()
		os_version = platform.version()
		self.logger.debug(f"Operating System: {os_name} {os_version}")

		python_version = platform.python_version()
		self.logger.debug(f"Python Version: {python_version}")

		pytorch_version = torch.__version__
		self.logger.debug(f"PyTorch Version: {pytorch_version}")

	def check_ffmpeg_installed(self):
		try:
			ffmpeg_version_output = subprocess.check_output(["ffmpeg", "-version"], text=True)
			first_line = ffmpeg_version_output.splitlines()[0]
			self.logger.debug(f"FFmpeg installed: {first_line}")
		except FileNotFoundError:
			self.logger.warning("FFmpeg is not installed. Please install FFmpeg to use this package.")

	def load_model(self):
		start_time = time()
		model, config = get_model_from_config(self.model_type, self.config_path)

		self.update_inference_params(config, self.inference_params)

		self.logger.info(f"Separator params: model_type: {self.model_type}, model_path: {self.model_path}, config_path: {self.config_path}, output_folder: {self.store_dirs}")
		self.logger.info(f"Audio params: output_format: {self.output_format}, audio_params: {self.audio_params}")
		self.logger.info(f"Model params: instruments: {config.training.get('instruments', None)}, target_instrument: {config.training.get('target_instrument', None)}")
		self.logger.debug(
			f"Model params: batch_size: {config.inference.get('batch_size', None)}, num_overlap: {config.inference.get('num_overlap', None)}, chunk_size: {config.audio.get('chunk_size', None)}, normalize: {config.inference.get('normalize', None)}, use_tta: {self.use_tta}"
		)

		if self.model_type in ["htdemucs", "apollo"]:
			state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
			if "state" in state_dict:
				state_dict = state_dict["state"]
			if "state_dict" in state_dict:
				state_dict = state_dict["state_dict"]
		else:
			state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
		model.load_state_dict(state_dict)

		if len(self.device_ids) > 1:
			model = torch.nn.DataParallel(model, device_ids=self.device_ids)
		model = model.to(self.device)
		model.eval()

		self.logger.debug(f"Loading model completed, duration: {time() - start_time:.2f} seconds")
		return model, config

	def normalize_audio(self, audio: np.ndarray):
		mono = audio.mean(0)
		mean, std = mono.mean(), mono.std()
		return (audio - mean) / std, {"mean": mean, "std": std}

	def apply_tta(self, mix: torch.Tensor, waveforms_orig: dict[str, torch.Tensor]):
		track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

		for i, augmented_mix in enumerate(track_proc_list):
			waveforms = demix(self.config, self.model, augmented_mix, self.device, model_type=self.model_type, callback=self.callback)
			for el in waveforms:
				if i == 0:
					waveforms_orig[el] += waveforms[el][::-1].copy()
				else:
					waveforms_orig[el] -= waveforms[el]
			self.logger.debug(f"TTA processing {i + 1}/{len(track_proc_list)} completed.")

		for el in waveforms_orig:
			waveforms_orig[el] /= len(track_proc_list) + 1

		return waveforms_orig

	def process_folder(self, input_folder):
		if not os.path.isdir(input_folder):
			raise ValueError(f"Input folder '{input_folder}' does not exist.")

		all_mixtures_path = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
		file_lists = all_mixtures_path.copy()

		sample_rate = getattr(self.config.audio, 'sample_rate', 44100)
		self.logger.info(f"Input_folder: {input_folder}, Total files found: {len(all_mixtures_path)}, Use sample rate: {sample_rate}")

		if not self.debug:
			all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")

		success_files = []
		for path in all_mixtures_path:
			if not self.debug:
				all_mixtures_path.set_postfix({"track": os.path.basename(path)})
			try:
				mix, sr = librosa.load(path, sr=sample_rate, mono=False)
			except Exception as e:
				self.logger.warning(f"Cannot process track: {path}, error: {str(e)}")
				continue

			self.logger.debug(f"Starting separation process for audio_file: {path}")

			if self.callback:
				self.callback["info"] = {"index": file_lists.index(path) + 1, "total": len(file_lists), "name": os.path.basename(path)}

			results = self.separate(mix)
			self.logger.debug(f"Separation audio_file: {path} completed. Starting to save results.")

			file_name, _ = os.path.splitext(os.path.basename(path))

			for instr in results.keys():
				save_dir = self.store_dirs.get(instr, "")
				if save_dir and type(save_dir) == str:
					os.makedirs(save_dir, exist_ok=True)
					self.save_audio(results[instr], sr, f"{file_name}_{instr}", save_dir)
					self.logger.debug(f"Saved {instr} for {file_name}_{instr}.{self.output_format} in {save_dir}")
				elif save_dir and type(save_dir) == list:
					for dir in save_dir:
						os.makedirs(dir, exist_ok=True)
						self.save_audio(results[instr], sr, f"{file_name}_{instr}", dir)
						self.logger.debug(f"Saved {instr} for {file_name}_{instr}.{self.output_format} in {dir}")

			success_files.append(os.path.basename(path))
			del mix, results
			gc.collect()
		return success_files

	def separate(self, mix):
		isstereo = True
		if self.model_type in ["bs_roformer", "mel_band_roformer"]:
			isstereo = self.config.model.get("stereo", True)

		if isstereo and len(mix.shape) == 1:  # if model is stereo, but track is mono, add a second channel
			mix = np.stack([mix, mix], axis=0)
			self.logger.warning(f"Track is mono, but model is stereo, adding a second channel.")
		elif isstereo and len(mix.shape) != 1 and mix.shape[0] > 2:  # fi model is stereo, but track has more than 2 channels, take mean
			mix = np.mean(mix, axis=0)
			mix = np.stack([mix, mix], axis=0)
			self.logger.warning(f"Track has more than 2 channels, taking mean of all channels and adding a second channel.")
		elif not isstereo and len(mix.shape) != 1:  # if model is mono, but track has more than 1 channels, take mean
			mix = np.mean(mix, axis=0)
			self.logger.warning(f"Track has more than 1 channels, but model is mono, taking mean of all channels.")

		instruments = self.config.training.instruments
		if self.config.training.target_instrument is not None:
			instruments = [self.config.training.target_instrument]
			self.logger.debug("Target instrument is not null, set primary_stem to target_instrument, secondary_stem will be calculated by mix - target_instrument")

		mix_orig = mix.copy()
		if 'normalize' in self.config.inference:
			if self.config.inference['normalize']:
				mix, norm_params = self.normalize_audio(mix)

		waveforms_orig = demix(self.config, self.model, mix, self.device, model_type=self.model_type, callback=self.callback)
		self.logger.debug(f"Finished demixing track, total instruments: {len(waveforms_orig)}")

		if self.use_tta:
			self.logger.debug("User needs to apply TTA, applying TTA to the waveforms.")
			waveforms_orig = self.apply_tta(mix, waveforms_orig)

		results = {}
		for instr in instruments:
			estimates = waveforms_orig[instr]
			if 'normalize' in self.config.inference:
				if self.config.inference['normalize']:
					estimates = estimates * norm_params["std"] + norm_params["mean"]

			results[instr] = estimates.T

		if self.config.training.target_instrument is not None:
			target_instrument = self.config.training.target_instrument
			other_instruments = [instr for instr in self.config.training.instruments if instr != target_instrument]

			self.logger.debug(f"target_instrument is not null, extracting instrumental from {target_instrument}, other_instruments: {other_instruments}")

			if other_instruments:
				estimates = mix_orig - waveforms_orig[target_instrument]
				if 'normalize' in self.config.inference:
					if self.config.inference['normalize']:
						estimates = estimates * norm_params["std"] + norm_params["mean"]

				results[other_instruments[0]] = estimates.T

		self.logger.debug("Separation process completed.")

		if self.callback:
			self.callback["progress"] = 1.0

		return results

	def save_audio(self, audio, sr, file_name, store_dir):
		if self.output_format.lower() == "flac":
			file = os.path.join(store_dir, file_name + ".flac")
			sf.write(file, audio, sr, subtype=self.audio_params["flac_bit_depth"])

		elif self.output_format.lower() == "mp3":
			file = os.path.join(store_dir, file_name + ".mp3")

			if audio.dtype != np.int16:
				audio = (audio * 32767).astype(np.int16)

			audio_segment = AudioSegment(audio.tobytes(), frame_rate=sr, sample_width=audio.dtype.itemsize, channels=2)

			audio_segment.export(file, format="mp3", bitrate=self.audio_params["mp3_bit_rate"])

		else:
			file = os.path.join(store_dir, file_name + ".wav")
			sf.write(file, audio, sr, subtype=self.audio_params["wav_bit_depth"])

	def del_cache(self):
		self.logger.debug("Running garbage collection...")
		gc.collect()
		if "mps" in self.device:
			self.logger.debug("Clearing MPS cache...")
			torch.mps.empty_cache()
		if "cuda" in self.device:
			self.logger.debug("Clearing CUDA cache...")
			torch.cuda.empty_cache()

	def update_inference_params(self, config, params):
		for key, value in {"batch_size": "inference", "num_overlap": "inference", "chunk_size": "audio", "normalize": "inference"}.items():
			if config[value].get(key) and params[key] is not None:
				config[value][key] = int(params[key]) if key != "normalize" else params[key]
		return config
