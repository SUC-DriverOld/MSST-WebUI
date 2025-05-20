import os
import shutil
import numpy as np
import soundfile as sf
import librosa
import glob
import traceback
from pydub import AudioSegment

from utils.logger import get_logger
from utils.constant import *
from webui.utils import load_configs, load_vr_model, load_msst_model, get_vr_model, get_msst_model
from inference.msst_infer import MSSeparator
from inference.vr_infer import VRSeparator
from utils.ensemble import ensemble_audios


class Presets:
	def __init__(self, presets, force_cpu=False, use_tta=False, logger=get_logger(), callback=None):
		self.presets = presets.get("flow", [])
		self.device = "auto" if not force_cpu else "cpu"
		self.force_cpu = force_cpu
		self.use_tta = use_tta
		self.logger = logger
		self.callback = callback
		self.total_steps = len(self.presets)
		self.preset_version = presets.get("version", "Unknown version")
		self.preset_name = presets.get("name", "Unknown name")

		if not self.is_exist_models()[0]:
			raise FileNotFoundError(f"Model not found: {self.is_exist_models()[1]}")

		webui_config = load_configs(WEBUI_CONFIG)
		self.debug = webui_config["settings"].get("debug", False)
		self.vr_model_path = webui_config["settings"]["uvr_model_dir"]
		self.batch_size = int(webui_config["inference"]["vr_batch_size"])
		self.window_size = int(webui_config["inference"]["vr_window_size"])
		self.aggression = int(webui_config["inference"]["vr_aggression"])
		self.enable_post_process = webui_config["inference"]["vr_enable_post_process"]
		self.post_process_threshold = float(webui_config["inference"]["vr_post_process_threshold"])
		self.high_end_process = webui_config["inference"]["vr_high_end_process"]
		self.wav_bit_depth = webui_config["settings"].get("wav_bit_depth", "FLOAT")
		self.flac_bit_depth = webui_config["settings"].get("flac_bit_depth", "PCM_24")
		self.mp3_bit_rate = webui_config["settings"].get("mp3_bit_rate", "320k")

		gpu_id = webui_config["inference"].get("device", None)
		self.gpu_ids = []
		if not self.force_cpu and gpu_id:
			try:
				for gpu in gpu_id:
					self.gpu_ids.append(int(gpu[: gpu.index(":")]))
			except:
				self.gpu_ids = [0]
		else:
			self.gpu_ids = [0]

	def get_step(self, step):
		return self.presets[step]

	def is_exist_models(self):
		for step in self.presets:
			model_name = step["model_name"]
			if model_name not in load_msst_model() and model_name not in load_vr_model():
				return False, model_name
		return True, None

	def msst_infer(self, model_type, config_path, model_path, input_folder, store_dict, output_format="wav"):
		separator = MSSeparator(
			model_type=model_type,
			config_path=config_path,
			model_path=model_path,
			device=self.device,
			device_ids=self.gpu_ids,
			output_format=output_format,
			use_tta=self.use_tta,
			store_dirs=store_dict,
			audio_params={"wav_bit_depth": self.wav_bit_depth, "flac_bit_depth": self.flac_bit_depth, "mp3_bit_rate": self.mp3_bit_rate},
			logger=self.logger,
			debug=self.debug,
			callback=self.callback,
		)
		separator.process_folder(input_folder)
		separator.del_cache()

	def vr_infer(self, model_name, input_folder, output_dir, output_format="wav"):
		separator = VRSeparator(
			logger=self.logger,
			debug=self.debug,
			model_file=os.path.join(self.vr_model_path, model_name),
			output_dir=output_dir,
			output_format=output_format,
			use_cpu=self.force_cpu,
			vr_params={
				"batch_size": self.batch_size,
				"window_size": self.window_size,
				"aggression": self.aggression,
				"enable_tta": self.use_tta,
				"enable_post_process": self.enable_post_process,
				"post_process_threshold": self.post_process_threshold,
				"high_end_process": self.high_end_process,
			},
			audio_params={"wav_bit_depth": self.wav_bit_depth, "flac_bit_depth": self.flac_bit_depth, "mp3_bit_rate": self.mp3_bit_rate},
			callback=self.callback,
		)
		separator.process_folder(input_folder)
		separator.del_cache()

	def process_folder(self):
		raise NotImplementedError("process_folder method is not implemented")


class PresetInfer(Presets):
	def __init__(self, presets, force_cpu=False, use_tta=False, logger=get_logger(), callback=None):
		super().__init__(presets, force_cpu, use_tta, logger, callback)

	def process_folder(self, input_folder, store_dir, output_format, extra_output=False):
		direct_output = os.path.join(store_dir, "extra_output") if extra_output else store_dir

		for current_step in range(self.total_steps):
			if current_step == 0:
				input_to_use = input_folder
				tmp_store_dir = os.path.join(TEMP_PATH, "step_1_output")
			if self.total_steps - 1 > current_step > 0:
				if input_to_use != input_folder:
					shutil.rmtree(input_to_use)
				input_to_use = tmp_store_dir
				tmp_store_dir = os.path.join(TEMP_PATH, f"step_{current_step + 1}_output")
			if current_step == self.total_steps - 1:
				input_to_use = tmp_store_dir
				tmp_store_dir = store_dir
			if self.total_steps == 1:
				input_to_use = input_folder
				tmp_store_dir = store_dir

			data = self.get_step(current_step)
			model_type = data["model_type"]
			model_name = data["model_name"]
			input_to_next = data["input_to_next"]
			output_to_storage = data["output_to_storage"]

			self.logger.info(f"\033[33mStep {current_step + 1}: Running inference using {model_name}\033[0m")
			if self.callback:
				self.callback["step_name"] = f"Step {current_step + 1}/{self.total_steps}: {model_name}"
				self.callback["info"] = {"index": -1, "total": -1, "name": ""}
				self.callback["progress"] = 0

			if model_type == "UVR_VR_Models":
				primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
				storage = {primary_stem: [], secondary_stem: []}
				storage[input_to_next].append(tmp_store_dir)
				for stem in output_to_storage:
					storage[stem].append(direct_output)

				self.logger.debug(f"input_to_next: {input_to_next}, output_to_storage: {output_to_storage}, storage: {storage}")
				self.vr_infer(model_name, input_to_use, storage, output_format)
			else:
				model_path, config_path, msst_model_type, _ = get_msst_model(model_name)
				stems = load_configs(config_path).training.get("instruments", [])
				storage = {stem: [] for stem in stems}
				storage[input_to_next].append(tmp_store_dir)
				for stem in output_to_storage:
					storage[stem].append(direct_output)

				self.logger.debug(f"input_to_next: {input_to_next}, output_to_storage: {output_to_storage}, storage: {storage}")
				self.msst_infer(msst_model_type, config_path, model_path, input_to_use, storage, output_format)

		if os.path.exists(TEMP_PATH):
			shutil.rmtree(TEMP_PATH)


class EnsembleInfer(Presets):
	def __init__(self, presets, force_cpu=False, use_tta=False, logger=get_logger(), callback=None):
		super().__init__(presets, force_cpu, use_tta, logger, callback)
		if self.total_steps < 2:
			raise ValueError("Ensemble inference requires at least 2 steps in the preset.")
		self.ensemble_data = None

	def ensemble(self, input_folder, store_dir, ensemble_mode, output_format="wav", extract_inst=False):
		assert self.ensemble_data is not None, "Ensemble data is not set. Please call process_folder first."
		success_files = []
		failed_files = []
		all_audios = os.listdir(input_folder)

		for audio in all_audios:
			if self.callback:
				self.callback["step_name"] = "Ensemble"
				self.callback["info"] = {"index": all_audios.index(audio) + 1, "total": len(all_audios), "name": audio}
				self.callback["progress"] = int((all_audios.index(audio) + 1) / len(all_audios))

			base_name = os.path.splitext(audio)[0]
			ensemble_audio = []
			ensemble_weights = []
			try:
				for model_name in self.ensemble_data.keys():
					audio_folder = self.ensemble_data[model_name]["store_dir"]
					audio_file = glob.glob(os.path.join(audio_folder, f"{base_name}*"))[0]
					ensemble_audio.append(audio_file)
					ensemble_weights.append(self.ensemble_data[model_name]["weight"])

				self.logger.debug(f"ensemble_audio: {ensemble_audio}, ensemble_weights: {ensemble_weights}")
				res, sr = ensemble_audios(ensemble_audio, ensemble_mode, ensemble_weights)
				save_filename = f"{base_name}_ensemble_{ensemble_mode}"
				save_audio(res, sr, output_format, save_filename, store_dir)

				if extract_inst:
					self.logger.debug(f"User choose to extract other instruments")
					raw, _ = librosa.load(os.path.join(input_folder, audio), sr=sr, mono=False)
					res = res.T

					if raw.shape[-1] != res.shape[-1]:
						self.logger.warning(f"Extracted audio shape: {res.shape} is not equal to raw audio shape: {raw.shape}, matching min length")
						min_length = min(raw.shape[-1], res.shape[-1])
						raw = raw[..., :min_length]
						res = res[..., :min_length]

					result = raw - res
					self.logger.debug(f"Extracted audio shape: {result.shape}")
					save_inst = f"{base_name}_ensemble_{ensemble_mode}_other"
					save_audio(result.T, sr, output_format, save_inst, store_dir)

				success_files.append(audio)

			except Exception as e:
				self.logger.error(f"Fail to ensemble audio: {audio}. Error: {e}\n{traceback.format_exc()}")
				failed_files.append(audio)
				continue

		if os.path.exists(TEMP_PATH):
			shutil.rmtree(TEMP_PATH)

		return success_files, failed_files

	def process_folder(self, input_folder):
		self.ensemble_data = dict()
		for i, data in enumerate(self.presets):
			model_type = data["model_type"]
			model_name = data["model_name"]
			stem = data["stem"]
			temp_store_dir = os.path.join(TEMP_PATH, model_name)
			self.ensemble_data[model_name] = {"store_dir": temp_store_dir, "weight": float(data["weight"])}

			self.logger.info(f"\033[33mStep {i + 1}: Running inference using {model_name}\033[0m")
			if self.callback:
				self.callback["step_name"] = f"Step {i + 1}/{self.total_steps}: {model_name}"
				self.callback["info"] = {"index": -1, "total": -1, "name": ""}
				self.callback["progress"] = 0

			if model_type == "UVR_VR_Models":
				storage = {stem: [temp_store_dir]}
				self.logger.debug(f"input_folder: {input_folder}, temp_store_dir: {temp_store_dir}, storage: {storage}")
				self.vr_infer(model_name, input_folder, storage)
			else:
				model_path, config_path, msst_model_type, _ = get_msst_model(model_name)
				storage = {stem: [temp_store_dir]}
				self.logger.debug(f"input_folder: {input_folder}, temp_store_dir: {temp_store_dir}, storage: {storage}")
				self.msst_infer(msst_model_type, config_path, model_path, input_folder, storage)


def save_audio(audio, sr, output_format, file_name, store_dir):
	webui_config = load_configs(WEBUI_CONFIG)
	wav_bit_depth = webui_config["settings"].get("wav_bit_depth", "FLOAT")
	flac_bit_depth = webui_config["settings"].get("flac_bit_depth", "PCM_24")
	mp3_bit_rate = webui_config["settings"].get("mp3_bit_rate", "320k")

	if output_format.lower() == "flac":
		file = os.path.join(store_dir, file_name + ".flac")
		sf.write(file, audio, sr, subtype=flac_bit_depth)
	elif output_format.lower() == "mp3":
		file = os.path.join(store_dir, file_name + ".mp3")
		if audio.dtype != np.int16:
			audio = (audio * 32767).astype(np.int16)
		audio_segment = AudioSegment(audio.tobytes(), frame_rate=sr, sample_width=audio.dtype.itemsize, channels=2)
		audio_segment.export(file, format="mp3", bitrate=mp3_bit_rate)
	else:
		file = os.path.join(store_dir, file_name + ".wav")
		sf.write(file, audio, sr, subtype=wav_bit_depth)
	return file
