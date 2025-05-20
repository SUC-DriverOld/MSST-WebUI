__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import shutil
import time
import gradio as gr
import multiprocessing
import traceback
import os

from utils.constant import *
from webui.utils import i18n, load_configs, save_configs, load_selected_model, logger, detailed_error
from webui.init import get_msst_model
from inference.msst_infer import MSSeparator


def save_model_config(selected_model, batch_size, num_overlap, chunk_size, normalize):
	_, config_path, _, _ = get_msst_model(selected_model)
	config = load_configs(config_path)

	if config.inference.get("batch_size"):
		config.inference["batch_size"] = int(batch_size)
	if config.inference.get("num_overlap"):
		config.inference["num_overlap"] = int(num_overlap)
	if config.audio.get("chunk_size"):
		config.audio["chunk_size"] = int(chunk_size)
	if config.inference.get("normalize"):
		config.inference["normalize"] = normalize
	save_configs(config, config_path)
	logger.debug(f"Saved model config: batch_size={batch_size}, num_overlap={num_overlap}, chunk_size={chunk_size}, normalize={normalize}")
	return i18n("配置保存成功!")


def reset_model_config(selected_model):
	_, original_config_path, _, _ = get_msst_model(selected_model)

	if original_config_path.startswith(UNOFFICIAL_MODEL):
		return i18n("非官方模型不支持重置配置!")

	dir_path, file_name = os.path.split(original_config_path)
	backup_dir_path = dir_path.replace("configs", "configs_backup", 1)
	backup_config_path = os.path.join(backup_dir_path, file_name)

	if os.path.exists(backup_config_path):
		shutil.copy(backup_config_path, original_config_path)
		update_inference_settings(selected_model)
		logger.debug(f"Reset model config: {backup_config_path} -> {original_config_path}")
		return i18n("配置重置成功!")
	else:
		return i18n("备份文件不存在!")


def update_inference_settings(selected_model):
	batch_size = gr.Slider(label="batch_size", interactive=False)
	num_overlap = gr.Slider(label="overlap", interactive=False)
	chunk_size = gr.Slider(label="chunk_size", interactive=False)
	normalize = gr.Checkbox(label="normalize", value=False, interactive=False)
	extract_instrumental = gr.CheckboxGroup(label=i18n("选择输出音轨"), interactive=False)

	if selected_model:
		_, config_path, _, _ = get_msst_model(selected_model)
		config = load_configs(config_path)

		if config.inference.get("batch_size"):
			batch_size = gr.Slider(label="batch_size", value=int(config.inference.get("batch_size")), interactive=True)
		if config.inference.get("num_overlap"):
			num_overlap = gr.Slider(label="overlap", value=int(config.inference.get("num_overlap")), interactive=True)
		if config.audio.get("chunk_size"):
			chunk_size = gr.Slider(label="chunk_size", value=int(config.audio.get("chunk_size")), interactive=True)
		if config.inference.get("normalize"):
			normalize = gr.Checkbox(label="normalize", value=config.inference.get("normalize"), interactive=True)
		extract_instrumental = gr.CheckboxGroup(label=i18n("选择输出音轨"), choices=config.training.get("instruments"), interactive=True)

	return batch_size, num_overlap, chunk_size, normalize, extract_instrumental


def update_selected_model(model_type):
	webui_config = load_configs(WEBUI_CONFIG)
	webui_config["inference"]["model_type"] = model_type
	save_configs(webui_config, WEBUI_CONFIG)
	return gr.Dropdown(label=i18n("选择模型"), choices=load_selected_model(), value=None, interactive=True, scale=4)


def save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
	config = load_configs(WEBUI_CONFIG)
	config["inference"]["selected_model"] = selected_model
	config["inference"]["device"] = gpu_id
	config["inference"]["output_format"] = output_format
	config["inference"]["force_cpu"] = force_cpu
	config["inference"]["instrumental"] = extract_instrumental
	config["inference"]["use_tta"] = use_tta
	config["inference"]["store_dir"] = store_dir
	config["inference"]["input_dir"] = input_folder
	save_configs(config, WEBUI_CONFIG)
	logger.debug(f"Saved MSST inference config: {config['inference']}")


def run_inference_single(selected_model, input_audio, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
	input_folder = None

	if not input_audio:
		return i18n("请上传至少一个音频文件!")
	if os.path.exists(TEMP_PATH):
		shutil.rmtree(TEMP_PATH)

	os.makedirs(TEMP_PATH)

	for audio in input_audio:
		shutil.copy(audio, TEMP_PATH)
	input_path = TEMP_PATH

	save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
	message = start_inference(selected_model, input_path, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
	shutil.rmtree(TEMP_PATH)
	return message


def run_multi_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
	save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
	return start_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)


def start_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
	if selected_model == "":
		return gr.Error(i18n("请选择模型"))
	if input_folder == "":
		return gr.Error(i18n("请选择输入目录"))
	if store_dir == "":
		return gr.Error(i18n("请选择输出目录"))

	gpu_ids = []
	if not force_cpu:
		if len(gpu_id) == 0:
			raise gr.Error(i18n("请选择GPU"))
		try:
			for gpu in gpu_id:
				gpu_ids.append(int(gpu[: gpu.index(":")]))
		except:
			gpu_ids = [0]
	else:
		gpu_ids = [0]

	gpu_ids = list(set(gpu_ids))
	device = "auto" if not force_cpu else "cpu"
	model_path, config_path, model_type, _ = get_msst_model(selected_model)
	webui_config = load_configs(WEBUI_CONFIG)
	debug = webui_config["settings"].get("debug", False)
	wav_bit_depth = webui_config["settings"].get("wav_bit_depth", "FLOAT")
	flac_bit_depth = webui_config["settings"].get("flac_bit_depth", "PCM_24")
	mp3_bit_rate = webui_config["settings"].get("mp3_bit_rate", "320k")

	if type(store_dir) == str:
		store_dict = {}
		model_config = load_configs(config_path)
		for inst in extract_instrumental:
			if inst in model_config.training.get("instruments"):  # bug of gr.CheckboxGroup, we must check if the instrument is in the model
				store_dict[inst] = store_dir
		if store_dict == {}:
			logger.warning(f"No selected instruments, extract all instruments to {store_dir}")
			store_dict = {k: store_dir for k in model_config.training.get("instruments")}
	else:
		store_dict = store_dir

	start_time = time.time()
	logger.info("Starting MSST inference process...")

	progress = gr.Progress()
	progress(0, desc="Starting", total=1, unit="percent")
	flag = (0, None)  # flag

	with multiprocessing.Manager() as manager:
		callback = manager.dict()
		callback["info"] = {"index": -1, "total": -1, "name": ""}
		callback["progress"] = 0  # percent
		callback["flag"] = flag  # flag

		msst_inference = multiprocessing.Process(
			target=run_inference,
			args=(model_type, config_path, model_path, device, gpu_ids, output_format, use_tta, store_dict, debug, wav_bit_depth, flac_bit_depth, mp3_bit_rate, input_folder, callback),
			name="msst_inference",
		)

		msst_inference.start()
		logger.debug(f"Inference process started, PID: {msst_inference.pid}")

		while msst_inference.is_alive():
			if callback["flag"][0]:
				break
			info = callback["info"]
			if info["index"] != -1:
				desc = f"{info['index']}/{info['total']}: {info['name']}"
				progress(callback["progress"], desc=desc, total=1, unit="percent")
			time.sleep(0.5)

		msst_inference.join()
		flag = callback["flag"]

	if flag[0]:
		if flag[0] == 1:
			return i18n("处理完成, 结果已保存至: ") + store_dir + i18n(", 耗时: ") + str(round(time.time() - start_time, 2)) + "s"
		elif flag[0] == -1:
			return i18n("处理失败: ") + detailed_error(flag[1])
	else:
		return i18n("进程意外终止")


def run_inference(model_type, config_path, model_path, device, gpu_ids, output_format, use_tta, store_dict, debug, wav_bit_depth, flac_bit_depth, mp3_bit_rate, input_folder, callback):
	logger.debug(
		f"Start MSST inference process with parameters: model_type={model_type}, config_path={config_path}, model_path={model_path}, device={device}, gpu_ids={gpu_ids}, output_format={output_format}, use_tta={use_tta}, store_dict={store_dict}, debug={debug}, wav_bit_depth={wav_bit_depth}, flac_bit_depth={flac_bit_depth}, mp3_bit_rate={mp3_bit_rate}, input_folder={input_folder}"
	)

	try:
		separator = MSSeparator(
			model_type=model_type,
			config_path=config_path,
			model_path=model_path,
			device=device,
			device_ids=gpu_ids,
			output_format=output_format,
			use_tta=use_tta,
			store_dirs=store_dict,
			audio_params={"wav_bit_depth": wav_bit_depth, "flac_bit_depth": flac_bit_depth, "mp3_bit_rate": mp3_bit_rate},
			logger=logger,
			debug=debug,
			callback=callback,
		)
		success_files = separator.process_folder(input_folder)
		separator.del_cache()

		logger.info(f"Successfully separated files: {success_files}")
		callback["flag"] = (1, success_files)
	except Exception as e:
		logger.error(f"Separation failed: {str(e)}\n{traceback.format_exc()}")
		callback["flag"] = (-1, str(e))


def stop_msst_inference():
	for process in multiprocessing.active_children():
		if process.name == "msst_inference":
			process.terminate()
			process.join()
			logger.info(f"Inference process stopped, PID: {process.pid}")
