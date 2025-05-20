__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import subprocess
import numpy as np
import librosa
import traceback
from pydub import AudioSegment
from tqdm import tqdm

from utils.constant import *
from webui.utils import i18n, load_configs, save_configs, logger
from tools.SOME.infer import infer


def convert_audio(uploaded_files, output_format, output_folder, sample_rate, channels, wav_bit_depth, flac_bit_depth, mp3_bit_rate, ogg_bit_rate):
	if not uploaded_files:
		return i18n("请上传至少一个文件")
	if channels == i18n("单声道"):
		channels = 1
	else:
		channels = 2

	ca = "pcm_s16le"
	sample_fmt = "s16"
	if wav_bit_depth == "PCM-16":
		ca = "pcm_s16le"
	elif wav_bit_depth == "PCM-24":
		ca = "pcm_s24le"
	elif wav_bit_depth == "PCM-32":
		ca = "pcm_s32le"
	if flac_bit_depth == "16-bit":
		sample_fmt = "s16"
	elif flac_bit_depth == "32-bit":
		sample_fmt = "s32"

	success_files = []
	fail_files = []
	os.makedirs(output_folder, exist_ok=True)

	logger.info(f"Converting audio files to {output_format} format. Output folder: {output_folder}. Total files: {len(uploaded_files)}")
	logger.info(f"Sample rate: {sample_rate}, Channels: {channels}, WAV bit depth: {wav_bit_depth}, FLAC bit depth: {flac_bit_depth}, MP3 bit rate: {mp3_bit_rate}, OGG bit rate: {ogg_bit_rate}")

	config = load_configs(WEBUI_CONFIG)
	config["tools"]["store_dir"] = output_folder
	config["tools"]["output_format"] = output_format
	config["tools"]["sample_rate"] = sample_rate
	config["tools"]["channels"] = channels
	config["tools"]["wav_bit_depth"] = wav_bit_depth
	config["tools"]["flac_bit_depth"] = flac_bit_depth
	config["tools"]["mp3_bit_rate"] = mp3_bit_rate
	config["tools"]["ogg_bit_rate"] = ogg_bit_rate
	save_configs(config, WEBUI_CONFIG)

	for file in tqdm(uploaded_files, desc="Converting audio files"):
		file_name = os.path.basename(file)
		basename = os.path.splitext(file_name)[0]

		if output_format == "wav":
			output_file = os.path.join(output_folder, f"{basename}_{sample_rate}_{wav_bit_depth}.wav")
			command = f'{FFMPEG} -i "{file}" -loglevel error -ar {sample_rate} -ac {channels} -c:a {ca} -y "{output_file}"'
		elif output_format == "flac":
			output_file = os.path.join(output_folder, f"{basename}_{sample_rate}_{flac_bit_depth}.flac")
			command = f'{FFMPEG} -i "{file}" -loglevel error -ar {sample_rate} -ac {channels} -sample_fmt {sample_fmt} -compression_level 5 -y "{output_file}"'
		elif output_format == "mp3":
			output_file = os.path.join(output_folder, f"{basename}_{sample_rate}_{mp3_bit_rate}.mp3")
			command = f'{FFMPEG} -i "{file}" -loglevel error -ar {sample_rate} -ac {channels} -b:a {mp3_bit_rate} -y "{output_file}"'
		elif output_format == "ogg":
			output_file = os.path.join(output_folder, f"{basename}_{sample_rate}_{ogg_bit_rate}.ogg")
			command = f'{FFMPEG} -i "{file}" -loglevel error -ar {sample_rate} -ac {channels} -b:a {ogg_bit_rate} -y "{output_file}"'
		else:
			output_file = os.path.join(output_folder, f"{basename}_{sample_rate}.{output_format}")
			command = f'{FFMPEG} -i "{file}" -loglevel error -ar {sample_rate} -ac {channels} -y "{output_file}"'

		try:
			subprocess.run(command, shell=False, check=True)
			success_files.append(output_file)
		except subprocess.CalledProcessError as e:
			logger.error(f"Fail to convert file: {file}, error: {e}\n{traceback.format_exc()}")
			fail_files.append(file)
			continue

	logger.info(f"Converted {len(success_files)} files successfully, failed to convert {len(fail_files)} files")
	return i18n("处理完成, 成功转换: ") + str(len(success_files)) + i18n("个文件, 失败: ") + str(len(fail_files)) + i18n("个文件") + f"\n{fail_files}"


def merge_audios(input_folder, output_folder):
	config = load_configs(WEBUI_CONFIG)
	config["tools"]["merge_audio_input"] = input_folder
	config["tools"]["store_dir"] = output_folder
	save_configs(config, WEBUI_CONFIG)

	combined_audio = AudioSegment.empty()
	os.makedirs(output_folder, exist_ok=True)
	output_file = os.path.join(output_folder, f"merged_audio_{os.path.basename(input_folder)}.wav")

	for filename in sorted(os.listdir(input_folder)):
		file_path = os.path.join(input_folder, filename)
		try:
			audio = AudioSegment.from_file(file_path)
			combined_audio += audio
		except Exception as e:
			logger.warning(f"Fail to merge file: {file_path}, skip it. Error: {e}")
			continue
	try:
		combined_audio.export(output_file, format="wav")
		logger.info(f"Merged audio files completed, saved as: {output_file}")
		return i18n("处理完成, 文件已保存为: ") + output_file
	except Exception as e:
		logger.error(f"Fail to export merged audio. Error: {e}\n{traceback.format_exc()}")
		return i18n("处理失败!") + str(e)


def caculate_sdr(reference_path, estimated_path):
	reference, _ = librosa.load(reference_path, sr=44100, mono=False)
	if reference.ndim == 1:
		reference = np.vstack((reference, reference))
	estimated, _ = librosa.load(estimated_path, sr=44100, mono=False)
	if estimated.ndim == 1:
		estimated = np.vstack((estimated, estimated))

	min_length = min(reference.shape[-1], estimated.shape[-1])
	reference = reference[..., :min_length]
	estimated = estimated[..., :min_length]

	def sdr(references, estimates):
		delta = 1e-7  # avoid numerical errors
		num = np.sum(np.square(references), axis=1)
		den = np.sum(np.square(references - estimates), axis=1)
		num += delta
		den += delta
		return 10 * np.log10(num / den)

	def si_sdr(reference, estimate):
		eps = 1e-07
		scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference**2 + eps, axis=1)
		scale = np.expand_dims(scale, axis=1)  # shape - [50, 1]
		reference = reference * scale
		sisdr = np.mean(10 * np.log10(np.sum(reference**2, axis=1) / (np.sum((reference - estimate) ** 2, axis=1) + eps) + eps))
		return sisdr

	sdr_value = sdr(reference, estimated)
	sisdr_value = si_sdr(reference, estimated)
	avg_sdr = np.mean(sdr_value)

	logger.info(f"References: {reference_path}, Estimates: {estimated_path}")
	logger.info(f"SDR: {sdr_value}, AVG-SDR: {avg_sdr}, SI-SDR: {sisdr_value}")
	return f"SDR: {sdr_value}\nAverage SDR: {avg_sdr}\nSI-SDR: {sisdr_value}"


def some_inference(audio_file, bpm, output_dir):
	if not os.path.isfile(SOME_WEIGHT):
		return i18n("请先下载SOME预处理模型并放置在tools/SOME_weights文件夹下! ")

	os.makedirs(output_dir, exist_ok=True)

	config = load_configs(WEBUI_CONFIG)
	config["tools"]["store_dir"] = output_dir
	save_configs(config, WEBUI_CONFIG)

	tempo = float(bpm)
	try:
		logger.info(f"Running SOME inference with audio file: {audio_file}, output dir: {output_dir}, tempo: {tempo}")
		midi = infer(SOME_WEIGHT, SOME_CONFIG, audio_file, output_dir, tempo)
		logger.info(f"SOME inference completed, MIDI file saved as: {midi}")
		return i18n("处理完成, 文件已保存为: ") + midi
	except Exception as e:
		logger.error(f"Fail to run SOME inference. Error: {e}\n{traceback.format_exc()}")
		return i18n("处理失败!") + str(e)
