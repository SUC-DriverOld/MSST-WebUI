__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr
import pandas as pd
import shutil
import time
import multiprocessing
import traceback

from utils.constant import *
from inference.preset_infer import PresetInfer
from webui.utils import i18n, load_configs, save_configs, get_vr_model, get_msst_model, logger, detailed_error


def get_presets_list() -> list:
	if os.path.exists(PRESETS):
		presets = [file for file in os.listdir(PRESETS) if file.endswith(".json")]
	else:
		presets = []
	return presets


def preset_backup_list():
	if not os.path.exists(PRESETS_BACKUP):
		return [i18n("暂无备份文件")]
	backup_files = [file for file in os.listdir(PRESETS_BACKUP) if file.endswith(".json")]
	return backup_files


def update_model_stem(model_type, model_name):
	if model_type == "UVR_VR_Models":
		primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
		input_to_next = gr.Radio(label=i18n("作为下一模型输入(或结果输出)的音轨"), choices=[primary_stem, secondary_stem], value=primary_stem, interactive=True)
		output_to_storage = gr.CheckboxGroup(label=i18n("直接保存至输出目录的音轨(可多选)"), choices=[i18n("不输出"), primary_stem, secondary_stem], interactive=True)
		return input_to_next, output_to_storage
	else:
		_, config_path, _, _ = get_msst_model(model_name)
		stems = load_configs(config_path).training.get("instruments", None)
		input_to_next = gr.Radio(label=i18n("作为下一模型输入(或结果输出)的音轨"), choices=stems, value=stems[0], interactive=True)
		output_to_storage = gr.CheckboxGroup(label=i18n("直接保存至输出目录的音轨(可多选)"), choices=[i18n("不输出")] + stems, interactive=True)
		return input_to_next, output_to_storage


def add_to_flow_func(model_type, model_name, input_to_next, output_to_storage, df):
	if not model_type or not model_name or not input_to_next:
		return df
	if not output_to_storage or i18n("不输出") in output_to_storage:
		output_to_storage = []
	elif model_type == "UVR_VR_Models":
		primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
		output_to_storage = [stem for stem in output_to_storage if stem in [primary_stem, secondary_stem]]
	else:
		_, config_path, _, _ = get_msst_model(model_name)
		stems = load_configs(config_path).training.get("instruments", None)
		output_to_storage = [stem for stem in output_to_storage if stem in stems]

	new_data = pd.DataFrame({"model_type": [model_type], "model_name": [model_name], "input_to_next": [input_to_next], "output_to_storage": [output_to_storage]})

	if df["model_type"].iloc[0] == "" or df["model_name"].iloc[0] == "" or df["input_to_next"].iloc[0] == "":
		return new_data

	updated_df = pd.concat([df, new_data], ignore_index=True)
	return updated_df


def save_flow_func(preset_name, df):
	if not preset_name:
		output_message = i18n("请填写预设名称")
		return output_message, None, None

	preset_dict = {"version": PRESET_VERSION, "name": preset_name, "flow": df.to_dict(orient="records")}
	os.makedirs(PRESETS, exist_ok=True)
	save_configs(preset_dict, os.path.join(PRESETS, f"{preset_name}.json"))

	output_message = i18n("预设") + preset_name + i18n("保存成功")
	preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
	preset_name_select = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())

	logger.info(f"Save preset: {preset_dict} as {preset_name}")
	return output_message, preset_name_delete, preset_name_select


def reset_flow_func():
	return gr.Dataframe(pd.DataFrame({"model_type": [""], "model_name": [""], "input_to_next": [""], "output_to_storage": [""]}), interactive=False, label=None)


def reset_last_func(df):
	if df.shape[0] == 1:
		return reset_flow_func()
	return df.iloc[:-1]


def load_preset(preset_name):
	if preset_name in os.listdir(PRESETS):
		preset_data = load_configs(os.path.join(PRESETS, preset_name))

		version = preset_data.get("version", "Unknown version")
		if version not in SUPPORTED_PRESET_VERSION:
			gr.Warning(i18n("不支持的预设版本: ") + str(version) + i18n(", 请重新制作预设。"))
			logger.error(f"Load preset: {preset_name} failed, unsupported version: {version}, supported version: {SUPPORTED_PRESET_VERSION}")
			return gr.Dataframe(
				pd.DataFrame(
					{"model_type": [i18n("预设版本不支持")], "model_name": [i18n("预设版本不支持")], "input_to_next": [i18n("预设版本不支持")], "output_to_storage": [i18n("预设版本不支持")]}
				),
				interactive=False,
				label=None,
			)

		preset_flow = pd.DataFrame({"model_type": [""], "model_name": [""], "input_to_next": [""], "output_to_storage": [""]})
		for step in preset_data["flow"]:
			preset_flow = add_to_flow_func(
				model_type=step["model_type"], model_name=step["model_name"], input_to_next=step["input_to_next"], output_to_storage=step["output_to_storage"], df=preset_flow
			)
		logger.info(f"Load preset: {preset_name}: {preset_data}")
		return preset_flow

	logger.error(f"Load preset: {preset_name} failed")
	return gr.Dataframe(
		pd.DataFrame({"model_type": [i18n("预设不存在")], "model_name": [i18n("预设不存在")], "input_to_next": [i18n("预设不存在")], "output_to_storage": [i18n("预设不存在")]}),
		interactive=False,
		label=None,
	)


def delete_func(preset_name):
	if preset_name in os.listdir(PRESETS):
		select_preset_backup = backup_preset_func(preset_name)
		os.remove(os.path.join(PRESETS, preset_name))
		output_message = i18n("预设") + preset_name + i18n("删除成功")
		preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
		preset_name_select = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
		preset_flow_delete = gr.Dataframe(
			pd.DataFrame({"model_type": [i18n("预设已删除")], "model_name": [i18n("预设已删除")], "input_to_next": [i18n("预设已删除")], "output_to_storage": [i18n("预设已删除")]}),
			interactive=False,
			label=None,
		)
		logger.info(f"Delete preset: {preset_name}")
		return output_message, preset_name_delete, preset_name_select, preset_flow_delete, select_preset_backup
	else:
		return i18n("预设不存在"), None, None, None, None


def backup_preset_func(preset_name):
	os.makedirs(PRESETS_BACKUP, exist_ok=True)
	backup_file = f"backup_{preset_name}"
	shutil.copy(os.path.join(PRESETS, preset_name), os.path.join(PRESETS_BACKUP, backup_file))
	logger.info(f"Backup preset: {preset_name} -> {backup_file}")
	return gr.Dropdown(label=i18n("选择需要恢复的预设流程备份"), choices=preset_backup_list(), interactive=True, scale=4)


def restore_preset_func(backup_file):
	backup_file_rename = backup_file
	if backup_file.startswith("backup_"):
		backup_file_rename = backup_file[7:]
	shutil.copy(os.path.join(PRESETS_BACKUP, backup_file), os.path.join(PRESETS, backup_file_rename))
	output_message_manage = i18n("已成功恢复备份") + backup_file
	preset_dropdown = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
	preset_name_delet = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list())
	preset_flow_delete = pd.DataFrame({"model_type": [i18n("请选择预设")], "model_name": [i18n("请选择预设")], "input_to_next": [i18n("请选择预设")], "output_to_storage": [i18n("请选择预设")]})
	logger.info(f"Restore preset: {backup_file} -> {backup_file_rename}")
	return output_message_manage, preset_dropdown, preset_name_delet, preset_flow_delete


def preset_inference_audio(input_audio, store_dir, preset, force_cpu, output_format, use_tta, extra_output_dir):
	if not input_audio:
		return i18n("请上传至少一个音频文件!")
	if os.path.exists(TEMP_PATH):
		shutil.rmtree(TEMP_PATH)
	os.makedirs(os.path.join(TEMP_PATH, "step_0_output"))

	for audio in input_audio:
		shutil.copy(audio, os.path.join(TEMP_PATH, "step_0_output"))
	input_folder = os.path.join(TEMP_PATH, "step_0_output")
	msg = preset_inference(input_folder, store_dir, preset, force_cpu, output_format, use_tta, extra_output_dir, is_audio=True)
	return msg


def preset_inference(input_folder, store_dir, preset_name, force_cpu, output_format, use_tta, extra_output_dir: bool, is_audio=False):
	if preset_name not in os.listdir(PRESETS):
		return i18n("预设") + preset_name + i18n("不存在")

	preset_data = load_configs(os.path.join(PRESETS, preset_name))
	preset_version = preset_data.get("version", "Unknown version")
	if preset_version not in SUPPORTED_PRESET_VERSION:
		logger.error(f"Unsupported preset version: {preset_version}, supported version: {SUPPORTED_PRESET_VERSION}")
		return i18n("不支持的预设版本: ") + preset_version + i18n(", 请重新制作预设。")

	config = load_configs(WEBUI_CONFIG)
	config["inference"]["preset"] = preset_name
	config["inference"]["force_cpu"] = force_cpu
	config["inference"]["output_format"] = output_format
	config["inference"]["preset_use_tta"] = use_tta
	config["inference"]["store_dir"] = store_dir
	config["inference"]["extra_output_dir"] = extra_output_dir
	if not is_audio:
		config["inference"]["input_dir"] = input_folder
	save_configs(config, WEBUI_CONFIG)

	if os.path.exists(TEMP_PATH) and not is_audio:
		shutil.rmtree(TEMP_PATH)

	start_time = time.time()
	logger.info(f"Starting preset inference process, use presets: {preset_name}")

	progress = gr.Progress()
	progress(0, desc="Starting", total=1, unit="percent")
	flag = (0, None)  # flag

	with multiprocessing.Manager() as manager:
		callback = manager.dict()
		callback["info"] = {"index": -1, "total": -1, "name": ""}
		callback["progress"] = 0  # percent
		callback["step_name"] = ""
		callback["flag"] = flag  # flag

		preset_inference = multiprocessing.Process(
			target=run_inference, name="preset_inference", args=(preset_data, force_cpu, use_tta, input_folder, store_dir, extra_output_dir, output_format, callback)
		)

		preset_inference.start()
		logger.debug(f"Inference process started, PID: {preset_inference.pid}")

		while preset_inference.is_alive():
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

		preset_inference.join()
		flag = callback["flag"]

	if flag[0]:
		if flag[0] == 1:
			logger.info(f"Successfully run preset {preset_name} inference. Cost time: {round(time.time() - start_time, 2)}s")
			return i18n("处理完成, 结果已保存至: ") + store_dir + i18n(", 耗时: ") + str(round(time.time() - start_time, 2)) + "s"
		elif flag[0] == -1:
			return i18n("处理失败: ") + detailed_error(flag[1])
	else:
		return i18n("进程意外终止")


def run_inference(preset_data, force_cpu, use_tta, input_folder, store_dir, extra_output_dir, output_format, callback):
	try:
		preset = PresetInfer(preset_data, force_cpu, use_tta, logger, callback)
		logger.debug(f"presets: {preset.presets}")
		logger.debug(f"total_steps: {preset.total_steps}, force_cpu: {force_cpu}, use_tta: {use_tta}, store_dir: {store_dir}, extra_output_dir: {extra_output_dir}, output_format: {output_format}")
		preset.process_folder(input_folder, store_dir, output_format, extra_output_dir)
		callback["flag"] = (1, "")
	except Exception as e:
		logger.error(f"Separation failed: {str(e)}\n{traceback.format_exc()}")
		callback["flag"] = (-1, str(e))


def stop_preset():
	for process in multiprocessing.active_children():
		if process.name == "preset_inference":
			process.terminate()
			process.join()
			logger.info(f"Inference process stopped, PID: {process.pid}")
