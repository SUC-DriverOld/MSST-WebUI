__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import shutil
import time
import gradio as gr
import multiprocessing
import traceback

from utils.constant import *
from webui.utils import i18n, get_vr_model, load_configs, save_configs, logger, detailed_error
from inference.vr_infer import VRSeparator


def load_vr_model_stem(model):
	primary_stem, secondary_stem, _, _ = get_vr_model(model)
	return (gr.Checkbox(label=f"{primary_stem} Only", value=False, interactive=True), gr.Checkbox(label=f"{secondary_stem} Only", value=False, interactive=True))


def save_vr_inference_config(
	vr_select_model,
	vr_window_size,
	vr_aggression,
	vr_output_format,
	vr_use_cpu,
	vr_primary_stem_only,
	vr_secondary_stem_only,
	vr_input,
	vr_store_dir,
	vr_batch_size,
	vr_post_process_threshold,
	vr_enable_tta,
	vr_high_end_process,
	vr_enable_post_process,
):
	config = load_configs(WEBUI_CONFIG)
	config["inference"]["vr_select_model"] = vr_select_model
	config["inference"]["vr_window_size"] = int(vr_window_size)
	config["inference"]["vr_aggression"] = int(vr_aggression)
	config["inference"]["output_format"] = vr_output_format
	config["inference"]["force_cpu"] = vr_use_cpu
	config["inference"]["vr_primary_stem_only"] = vr_primary_stem_only
	config["inference"]["vr_secondary_stem_only"] = vr_secondary_stem_only
	config["inference"]["input_dir"] = vr_input
	config["inference"]["store_dir"] = vr_store_dir
	config["inference"]["vr_batch_size"] = int(vr_batch_size)
	config["inference"]["vr_post_process_threshold"] = float(vr_post_process_threshold)
	config["inference"]["vr_enable_tta"] = vr_enable_tta
	config["inference"]["vr_high_end_process"] = vr_high_end_process
	config["inference"]["vr_enable_post_process"] = vr_enable_post_process
	save_configs(config, WEBUI_CONFIG)
	logger.debug(f"Saved VR inference config: {config['inference']}")


def vr_inference_single(
	vr_select_model,
	vr_window_size,
	vr_aggression,
	vr_output_format,
	vr_use_cpu,
	vr_primary_stem_only,
	vr_secondary_stem_only,
	audio_input,
	vr_store_dir,
	vr_batch_size,
	vr_post_process_threshold,
	vr_enable_tta,
	vr_high_end_process,
	vr_enable_post_process,
):
	vr_input_save = None

	if not audio_input:
		return i18n("请上传至少一个音频文件!")
	if os.path.exists(TEMP_PATH):
		shutil.rmtree(TEMP_PATH)

	os.makedirs(TEMP_PATH)

	for audio in audio_input:
		shutil.copy(audio, TEMP_PATH)
	vr_audio_input = TEMP_PATH

	save_vr_inference_config(
		vr_select_model,
		vr_window_size,
		vr_aggression,
		vr_output_format,
		vr_use_cpu,
		vr_primary_stem_only,
		vr_secondary_stem_only,
		vr_input_save,
		vr_store_dir,
		vr_batch_size,
		vr_post_process_threshold,
		vr_enable_tta,
		vr_high_end_process,
		vr_enable_post_process,
	)
	message = start_inference(
		vr_select_model,
		vr_window_size,
		vr_aggression,
		vr_output_format,
		vr_use_cpu,
		vr_primary_stem_only,
		vr_secondary_stem_only,
		vr_audio_input,
		vr_store_dir,
		vr_batch_size,
		vr_post_process_threshold,
		vr_enable_tta,
		vr_high_end_process,
		vr_enable_post_process,
	)
	shutil.rmtree(TEMP_PATH)
	return message


def vr_inference_multi(
	vr_select_model,
	vr_window_size,
	vr_aggression,
	vr_output_format,
	vr_use_cpu,
	vr_primary_stem_only,
	vr_secondary_stem_only,
	folder_input,
	vr_store_dir,
	vr_batch_size,
	vr_post_process_threshold,
	vr_enable_tta,
	vr_high_end_process,
	vr_enable_post_process,
):
	save_vr_inference_config(
		vr_select_model,
		vr_window_size,
		vr_aggression,
		vr_output_format,
		vr_use_cpu,
		vr_primary_stem_only,
		vr_secondary_stem_only,
		folder_input,
		vr_store_dir,
		vr_batch_size,
		vr_post_process_threshold,
		vr_enable_tta,
		vr_high_end_process,
		vr_enable_post_process,
	)
	return start_inference(
		vr_select_model,
		vr_window_size,
		vr_aggression,
		vr_output_format,
		vr_use_cpu,
		vr_primary_stem_only,
		vr_secondary_stem_only,
		folder_input,
		vr_store_dir,
		vr_batch_size,
		vr_post_process_threshold,
		vr_enable_tta,
		vr_high_end_process,
		vr_enable_post_process,
	)


def start_inference(
	vr_select_model,
	vr_window_size,
	vr_aggression,
	vr_output_format,
	vr_use_cpu,
	vr_primary_stem_only,
	vr_secondary_stem_only,
	audio_input,
	vr_store_dir,
	vr_batch_size,
	vr_post_process_threshold,
	vr_enable_tta,
	vr_high_end_process,
	vr_enable_post_process,
):
	if not audio_input:
		return i18n("请上传至少一个音频文件!")
	if not vr_select_model:
		return i18n("请选择模型")
	if not vr_store_dir:
		return i18n("请选择输出目录")

	webui_config = load_configs(WEBUI_CONFIG)
	debug = webui_config["settings"].get("debug", False)
	wav_bit_depth = webui_config["settings"].get("wav_bit_depth", "FLOAT")
	flac_bit_depth = webui_config["settings"].get("flac_bit_depth", "PCM_24")
	mp3_bit_rate = webui_config["settings"].get("mp3_bit_rate", "320k")
	primary_stem, secondary_stem, _, model_path = get_vr_model(vr_select_model)
	model_file = os.path.join(model_path, vr_select_model)

	if not vr_primary_stem_only and not vr_secondary_stem_only:
		logger.warning("Both primary and secondary stem are disabled, enable both by default")
		vr_primary_stem_only = True
		vr_secondary_stem_only = True

	output_dir = {primary_stem: vr_store_dir if vr_primary_stem_only else "", secondary_stem: vr_store_dir if vr_secondary_stem_only else ""}

	start_time = time.time()
	logger.info("Straring VR inference process...")

	progress = gr.Progress()
	progress(0, desc="Starting", total=1, unit="percent")
	flag = (0, None)  # flag

	with multiprocessing.Manager() as manager:
		callback = manager.dict()
		callback["info"] = {"index": -1, "total": -1, "name": ""}
		callback["progress"] = 0  # percent
		callback["flag"] = flag  # flag

		vr_inference = multiprocessing.Process(
			target=run_inference,
			args=(
				debug,
				model_file,
				output_dir,
				vr_output_format,
				vr_use_cpu,
				int(vr_batch_size),
				int(vr_window_size),
				int(vr_aggression),
				vr_enable_tta,
				vr_enable_post_process,
				vr_post_process_threshold,
				vr_high_end_process,
				wav_bit_depth,
				flac_bit_depth,
				mp3_bit_rate,
				audio_input,
				callback,
			),
			name="vr_inference",
		)

		vr_inference.start()
		logger.debug(f"Inference process started, PID: {vr_inference.pid}")

		while vr_inference.is_alive():
			if callback["flag"][0]:
				break
			info = callback["info"]
			if info["index"] != -1:
				desc = f"{info['index']}/{info['total']}: {info['name']}"
				progress(callback["progress"], desc=desc, total=1, unit="percent")
			time.sleep(0.5)

		vr_inference.join()
		flag = callback["flag"]

	if flag[0]:
		if flag[0] == 1:
			return i18n("处理完成, 结果已保存至: ") + vr_store_dir + i18n(", 耗时: ") + str(round(time.time() - start_time, 2)) + "s"
		elif flag[0] == -1:
			return i18n("处理失败: ") + detailed_error(flag[1])
	else:
		return i18n("进程意外终止")


def run_inference(
	debug,
	model_file,
	output_dir,
	output_format,
	use_cpu,
	batch_size,
	window_size,
	aggression,
	enable_tta,
	enable_post_process,
	post_process_threshold,
	high_end_process,
	wav_bit_depth,
	flac_bit_depth,
	mp3_bit_rate,
	input_folder,
	callback,
):
	logger.debug(
		f"Start VR inference process with parameters: debug={debug}, model_file={model_file}, output_dir={output_dir}, output_format={output_format}, use_cpu={use_cpu}, batch_size={batch_size}, window_size={window_size}, aggression={aggression}, enable_tta={enable_tta}, enable_post_process={enable_post_process}, post_process_threshold={post_process_threshold}, high_end_process={high_end_process}, wav_bit_depth={wav_bit_depth}, flac_bit_depth={flac_bit_depth}, mp3_bit_rate={mp3_bit_rate}, input_folder={input_folder}"
	)

	try:
		separator = VRSeparator(
			logger=logger,
			debug=debug,
			model_file=model_file,
			output_dir=output_dir,
			output_format=output_format,
			use_cpu=use_cpu,
			vr_params={
				"batch_size": batch_size,
				"window_size": window_size,
				"aggression": aggression,
				"enable_tta": enable_tta,
				"enable_post_process": enable_post_process,
				"post_process_threshold": post_process_threshold,
				"high_end_process": high_end_process,
			},
			audio_params={"wav_bit_depth": wav_bit_depth, "flac_bit_depth": flac_bit_depth, "mp3_bit_rate": mp3_bit_rate},
			callback=callback,
		)
		success_files = separator.process_folder(input_folder)
		separator.del_cache()

		logger.info(f"Successfully separated files: {success_files}")
		callback["flag"] = (1, success_files)
	except Exception as e:
		logger.error(f"Separation failed: {str(e)}\n{traceback.format_exc()}")
		callback["flag"] = (-1, str(e))


def stop_vr_inference():
	for process in multiprocessing.active_children():
		if process.name == "vr_inference":
			process.terminate()
			process.join()
			logger.info(f"Inference process stopped, PID: {process.pid}")
