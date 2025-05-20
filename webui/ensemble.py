__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr
import pandas as pd
import traceback
import shutil
import time
import multiprocessing

from utils.constant import *
from utils.ensemble import ensemble_audios
from inference.preset_infer import EnsembleInfer, save_audio
from webui.utils import i18n, load_configs, save_configs, get_vr_model, get_msst_model, logger, detailed_error


def update_model_stem(model_type, model_name):
	if model_type == "UVR_VR_Models":
		primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
		output_stems = gr.Radio(label=i18n("输出音轨"), choices=[primary_stem, secondary_stem], value=primary_stem, interactive=True)
		return output_stems
	else:
		_, config_path, _, _ = get_msst_model(model_name)
		stems = load_configs(config_path).training.get("instruments", None)
		output_stems = gr.Radio(label=i18n("输出音轨"), choices=stems, value=stems[0], interactive=True)
		return output_stems


def add_to_ensemble_flow(model_type, model_name, stem_weight, stem, df):
	if not model_type or not model_name or not stem_weight or not stem:
		return df
	new_data = pd.DataFrame({"model_type": [model_type], "model_name": [model_name], "stem": [stem], "weight": [stem_weight]})
	if df["model_type"].iloc[0] == "" or df["model_name"].iloc[0] == "" or df["stem"].iloc[0] == "" or df["weight"].iloc[0] == "":
		return new_data
	updated_df = pd.concat([df, new_data], ignore_index=True)
	return updated_df


def reset_flow_func():
	return gr.Dataframe(pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""], "weight": [""]}), interactive=False, label=None)


def reset_last_func(df):
	if df.shape[0] == 1:
		return reset_flow_func()
	return df.iloc[:-1]


def save_ensemble_preset_func(df):
	if df.shape[0] < 2:
		raise gr.Error(i18n("请至少添加2个模型到合奏流程"))
	config = load_configs(WEBUI_CONFIG)
	config["inference"]["ensemble_preset"] = {"flow": df.to_dict(orient="records")}
	save_configs(config, WEBUI_CONFIG)
	logger.info(f"Ensemble flow saved: {df.to_dict(orient='records')}")
	gr.Info(i18n("合奏流程已保存"))


def load_ensemble():
	flow = pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""], "weight": [""]})
	try:
		config = load_configs(WEBUI_CONFIG)
		data = config["inference"]["ensemble_preset"]
		if not data:
			return flow
		for step in data["flow"]:
			flow = add_to_ensemble_flow(model_type=step["model_type"], model_name=step["model_name"], stem_weight=step["weight"], stem=step["stem"], df=flow)
		return flow
	except:
		return flow


def inference_audio_func(ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_audio, extract_inst):
	if not input_audio:
		return i18n("请上传至少一个音频文件!")
	if os.path.exists(TEMP_PATH):
		shutil.rmtree(TEMP_PATH)
	os.makedirs(os.path.join(TEMP_PATH, "ensemble_raw"))

	for audio in input_audio:
		shutil.copy(audio, os.path.join(TEMP_PATH, "ensemble_raw"))
	input_folder = os.path.join(TEMP_PATH, "ensemble_raw")
	msg = inference_folder_func(ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_folder, extract_inst, is_audio=True)
	return msg


def inference_folder_func(ensemble_mode, output_format, force_cpu, use_tta, store_dir, input_folder, extract_inst, is_audio=False):
	config = load_configs(WEBUI_CONFIG)
	preset_data = config["inference"]["ensemble_preset"]
	if not preset_data:
		return i18n("请先创建合奏流程")

	config["inference"]["force_cpu"] = force_cpu
	config["inference"]["output_format"] = output_format
	config["inference"]["store_dir"] = store_dir
	config["inference"]["ensemble_use_tta"] = use_tta
	config["inference"]["ensemble_type"] = ensemble_mode
	config["inference"]["ensemble_extract_inst"] = extract_inst

	if not is_audio:
		config["inference"]["input_dir"] = input_folder
	save_configs(config, WEBUI_CONFIG)

	if os.path.exists(TEMP_PATH) and not is_audio:
		shutil.rmtree(TEMP_PATH)

	start_time = time.time()
	logger.info("Starting ensemble inference process")

	progress = gr.Progress()
	progress(0, desc="Starting", total=1, unit="percent")
	flag = (0, None)  # flag

	with multiprocessing.Manager() as manager:
		callback = manager.dict()
		callback["info"] = {"index": -1, "total": -1, "name": ""}
		callback["progress"] = 0  # percent
		callback["step_name"] = ""
		callback["flag"] = flag  # flag

		ensemble_inference = multiprocessing.Process(
			target=run_inference, args=(preset_data, force_cpu, use_tta, store_dir, input_folder, ensemble_mode, output_format, extract_inst, callback), name="ensemble_inference"
		)

		ensemble_inference.start()
		logger.debug(f"Inference process started, PID: {ensemble_inference.pid}")

		while ensemble_inference.is_alive():
			if callback["flag"][0]:
				break
			desc = ""
			if callback["step_name"]:
				desc += callback["step_name"] + " "
			info = callback["info"]
			if info["index"] != -1:
				desc += f"{info['index']}/{info['total']}: {info['name']}"
			else:
				desc += "Strarting"
			progress(callback["progress"], desc=desc, total=1, unit="percent")
			time.sleep(0.5)

		ensemble_inference.join()
		flag = callback["flag"]

	if flag[0]:
		if flag[0] == 1:
			logger.info(f"Successfully run ensemble inference. Cost time: {round(time.time() - start_time, 2)}s")
			return (
				i18n("处理完成, 成功: ")
				+ str(len(flag[1][0]))
				+ i18n("个文件, 失败: ")
				+ str(len(flag[1][1]))
				+ i18n("个文件")
				+ i18n(", 结果已保存至: ")
				+ store_dir
				+ i18n(", 耗时: ")
				+ str(round(time.time() - start_time, 2))
				+ "s"
			)
		elif flag[0] == -1:
			return i18n("处理失败: ") + detailed_error(flag[1])
	else:
		return i18n("进程意外终止")


def run_inference(preset_data, force_cpu, use_tta, store_dir, input_folder, ensemble_mode, output_format, extract_inst, callback):
	try:
		preset = EnsembleInfer(preset_data, force_cpu, use_tta, logger, callback)
		logger.debug(f"presets: {preset.presets}")
		logger.debug(f"total_models: {preset.total_steps}, force_cpu: {force_cpu}, use_tta: {use_tta}, store_dir: {store_dir}, output_format: {output_format}")
		preset.process_folder(input_folder)
		results = preset.ensemble(input_folder, store_dir, ensemble_mode, output_format, extract_inst)
		callback["flag"] = (1, results)
	except Exception as e:
		logger.error(f"Separation failed: {str(e)}\n{traceback.format_exc()}")
		callback["flag"] = (-1, str(e))


def stop_ensemble_func():
	for process in multiprocessing.active_children():
		if process.name == "ensemble_inference":
			process.terminate()
			process.join()
			logger.info(f"Inference process stopped, PID: {process.pid}")


def ensemble_files(files, ensemble_mode, weights, output_path, output_format):
	if len(files) < 2:
		return i18n("请上传至少2个文件")
	if len(files) != len(weights.split()):
		return i18n("上传的文件数目与权重数目不匹配")

	config = load_configs(WEBUI_CONFIG)
	config["inference"]["ensemble_type"] = ensemble_mode
	config["inference"]["store_dir"] = output_path
	save_configs(config, WEBUI_CONFIG)

	os.makedirs(output_path, exist_ok=True)
	weights = [float(w) for w in weights.split()]
	filename = f"ensemble_{ensemble_mode}_{len(files)}_songs"
	try:
		res, sr = ensemble_audios(files, ensemble_mode, weights)
		file = save_audio(res, sr, output_format, filename, output_path)
		logger.info(f"Ensemble files completed, saved to: {file}")
		return i18n("处理完成, 文件已保存为: ") + file
	except Exception as e:
		logger.error(f"Fail to ensemble files. Error: {e}\n{traceback.format_exc()}")
		return i18n("处理失败!") + str(e)
