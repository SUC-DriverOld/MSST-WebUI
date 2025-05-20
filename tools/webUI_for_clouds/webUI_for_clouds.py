__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr
import pandas as pd
import platform

from torch import cuda
from multiprocessing import cpu_count
from utils.constant import *
from webui.utils import i18n, load_configs, webui_restart, log_level_debug, change_to_audio_infer, change_to_folder_infer, logger
from tools.webUI_for_clouds.download_models import download_model

webui_config = load_configs(WEBUI_CONFIG)
language_dict = load_configs(LANGUAGE)
device = []
force_cpu_value = False


def msst_cloud_model(model_type):
	# if not model_type:
	#     return None
	# model_map = load_configs(MSST_MODEL)
	# model_list = []
	# for model in model_map[model_type]:
	#     model_list.append(model["name"])
	# return model_list
	msst_config = load_configs(MODELS_INFO)
	msst_models = []
	for key, model in msst_config.items():
		if model["model_class"] == model_type:
			msst_models.append(key)
	return msst_models


def load_vr_cloud_model():
	model_map = load_configs(MODELS_INFO)
	list = []
	for key, model in model_map.items():
		if model["model_class"] == "VR_Models":
			list.append(key)
	return list


def load_msst_cloud_model(model_type):
	model_list = msst_cloud_model(model_type)
	return gr.Dropdown(label=i18n("选择模型"), choices=model_list, value=None, interactive=True, scale=4)


def load_preset_cloud_model(model_type):
	if model_type == "UVR_VR_Models":
		model_map = load_vr_cloud_model()
		return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)
	else:
		model_map = msst_cloud_model(model_type)
		return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)


def cloud_msst_infer_audio(selected_model, input_audio, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
	from webui.msst import run_inference_single

	if selected_model:
		assert download_model("msst", selected_model), i18n("模型下载失败, 请重试!")
		return run_inference_single(selected_model, input_audio, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)


def cloud_msst_infer_folder(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
	from webui.msst import run_multi_inference

	if selected_model:
		assert download_model("msst", selected_model), i18n("模型下载失败, 请重试!")
		return run_multi_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)


def cloud_vr_infer_audio(
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
	from webui.vr import vr_inference_single

	if vr_select_model:
		assert download_model("uvr", vr_select_model), i18n("模型下载失败, 请重试!")
		return vr_inference_single(
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
		)


def cloud_vr_infer_folder(
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
	from webui.vr import vr_inference_multi

	if vr_select_model:
		assert download_model("uvr", vr_select_model), i18n("模型下载失败, 请重试!")
		return vr_inference_multi(
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


def check_preset(preset_data):
	for model in preset_data["flow"]:
		model_type = model["model_type"]
		model_name = model["model_name"]
		if model_type == "UVR_VR_Models":
			assert download_model("uvr", model_name), i18n("模型下载失败, 请重试!")
		else:
			assert download_model("msst", model_name), i18n("模型下载失败, 请重试!")
	return True


def cloud_preset_infer_audio(input_audio, store_dir, preset, force_cpu, output_format, use_tta, extra_output_dir):
	from webui.preset import preset_inference_audio

	preset_data = load_configs(os.path.join(PRESETS, preset))
	assert check_preset(preset_data), i18n("模型下载失败, 请重试!")
	return preset_inference_audio(input_audio, store_dir, preset, force_cpu, output_format, use_tta, extra_output_dir)


def cloud_preset_infer_folder(input_folder, store_dir, preset_name, force_cpu, output_format, use_tta, extra_output_dir):
	from webui.preset import preset_inference

	preset_data = load_configs(os.path.join(PRESETS, preset_name))
	assert check_preset(preset_data), i18n("模型下载失败, 请重试!")
	return preset_inference(input_folder, store_dir, preset_name, force_cpu, output_format, use_tta, extra_output_dir)


def cloud_ensemble_infer_audio(ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_audio, extract_inst):
	from webui.ensemble import inference_audio_func

	ensemble_data = webui_config["inference"]["ensemble_preset"]
	assert check_preset(ensemble_data), i18n("模型下载失败, 请重试!")
	return inference_audio_func(ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_audio, extract_inst)


def cloud_ensemble_infer_folder(ensemble_mode, output_format, force_cpu, use_tta, store_dir, input_folder, extract_inst):
	from webui.ensemble import inference_folder_func

	ensemble_data = webui_config["inference"]["ensemble_preset"]
	assert check_preset(ensemble_data), i18n("模型下载失败, 请重试!")
	return inference_folder_func(ensemble_mode, output_format, force_cpu, use_tta, store_dir, input_folder, extract_inst)


def launch(server_name=None, server_port=None, share=True):
	global device, force_cpu_value

	os.makedirs("input", exist_ok=True)
	os.makedirs("results", exist_ok=True)

	debug = webui_config["settings"].get("debug", False)
	if debug:
		log_level_debug(True)
	else:
		log_level_debug(False)

	devices = {}
	force_cpu = False
	if cuda.is_available():
		for i in range(cuda.device_count()):
			devices[f"cuda{i}"] = f"{i}: {cuda.get_device_name(i)}"
		logger.info(i18n("检测到CUDA, 设备信息: ") + str(devices))
	else:
		devices = {"cpu": i18n("无可用的加速设备, 使用CPU")}
		logger.warning(i18n("\033[33m未检测到可用的加速设备, 使用CPU\033[0m"))
		force_cpu = True
	device = [value for _, value in devices.items()]
	force_cpu_value = True if (webui_config["inference"]["force_cpu"] or force_cpu) else False

	logger.info(f"WebUI Version: {PACKAGE_VERSION}, System: {platform.system()}, Machine: {platform.machine()}")
	app().launch(share=share, show_api=False, server_name=server_name, server_port=server_port)


def app():
	with gr.Blocks(theme=gr.Theme.load("tools/themes/theme_blue.json")) as webui:
		gr.Markdown(value=f"""### Music-Source-Separation-Training-Inference-Webui For Clouds v{PACKAGE_VERSION}""")
		gr.Markdown(
			value=i18n(
				"作者: [Github@KitsuneX07](https://github.com/KitsuneX07) | [Github@SUC-DriverOld](https://github.com/SUC-DriverOld), [点击前往教程文档](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)"
			)
		)
		gr.Markdown(value=i18n("**请将需要处理的音频放置到input文件夹内, 处理完成后的音频将会保存到results文件夹内! 云端输入输出目录不可更改!**"))

		with gr.Tabs():
			with gr.TabItem(label=i18n("文件管理")):
				files()
			with gr.TabItem(label=i18n("MSST分离")):
				msst()
			with gr.TabItem(label=i18n("UVR分离")):
				vr()
			with gr.TabItem(label=i18n("预设流程")):
				preset()
			with gr.TabItem(label=i18n("合奏模式")):
				ensemble()
			with gr.TabItem(label=i18n("小工具")):
				tools()
			with gr.TabItem(label=i18n("MSST训练")):
				train()
			with gr.TabItem(label=i18n("设置")):
				settings()
	return webui


def files():
	from webui.file_manager import delete_input_files, delete_results_files, reflash_files, download_results_files, upload_files_to_input

	gr.Markdown(
		value=i18n(
			"文件管理页面是云端WebUI特有的页面, 用于上传, 下载, 删除文件。<br>1. 上传文件: 将文件上传到input文件夹内。可以勾选是否自动解压zip文件<br>2. 下载文件: 以zip格式打包results文件夹内的文件, 输出至WebUI以供下载。注意: 打包不会删除results文件夹, 若打包后不再需要分离结果, 请点击按钮手动删除。<br>3. 删除文件: 删除input和results文件夹内的文件。"
		)
	)
	with gr.Row():
		with gr.Column():
			with gr.Row():
				delete_input = gr.Button(i18n("删除input文件夹内所有文件"), variant="primary")
				delete_results = gr.Button(i18n("删除results文件夹内所有文件"), variant="primary")
			reflash = gr.Button(i18n("刷新input和results文件列表"), variant="primary")
			download_results = gr.Button(i18n("打包results文件夹内所有文件"), variant="primary")
			upload_files = gr.Files(label=i18n("上传一个或多个文件至input文件夹"), type="filepath")
			auto_unzip = gr.Checkbox(label=i18n("自动解压zip文件(仅支持zip, 压缩包内文件名若含有非ASCII字符, 解压后文件名可能为乱码)"), value=True, interactive=True)
			upload_button = gr.Button(i18n("上传文件"), variant="primary")
		with gr.Column():
			file_lists = gr.Textbox(label=i18n("input和results文件列表"), value=i18n("请先点击刷新按钮"), lines=4, interactive=False)
			results_zip = gr.File(label=i18n("下载results文件夹内所有文件"), type="filepath", interactive=False)

	delete_input.click(fn=delete_input_files, outputs=file_lists)
	delete_results.click(fn=delete_results_files, outputs=file_lists)
	reflash.click(fn=reflash_files, outputs=file_lists)
	download_results.click(fn=download_results_files, outputs=results_zip)
	upload_button.click(fn=upload_files_to_input, inputs=[upload_files, auto_unzip], outputs=file_lists)


def msst():
	from webui.msst import stop_msst_inference, save_model_config, reset_model_config, update_inference_settings

	gr.Markdown(value=i18n("MSST音频分离原项目地址: [https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)"))
	with gr.Row():
		select_model_type = gr.Dropdown(label=i18n("选择模型类型"), choices=["vocal_models", "multi_stem_models", "single_stem_models"], value=None, interactive=True, scale=1)
		selected_model = gr.Dropdown(label=i18n("选择模型"), choices=None, value=None, interactive=True, scale=4)
	with gr.Row():
		gpu_id = gr.CheckboxGroup(label=i18n("选择使用的GPU"), choices=device, value=device[0], interactive=True)
		output_format = gr.Radio(label=i18n("输出格式"), choices=["wav", "mp3", "flac"], value="wav", interactive=True)
	with gr.Row():
		extract_instrumental = gr.CheckboxGroup(label=i18n("选择输出音轨"), choices=[i18n("请先选择模型")], value=None, interactive=True)
		force_cpu = gr.Checkbox(info=i18n("强制使用CPU推理, 注意: 使用CPU推理速度非常慢!"), label=i18n("使用CPU"), value=force_cpu_value, interactive=True)
	with gr.Tabs():
		with gr.TabItem(label=i18n("输入文件夹")) as folder_tab:
			folder_input = gr.Textbox(label=i18n("输入目录"), value="input/", interactive=False, scale=4)
		with gr.TabItem(label=i18n("输入音频")) as audio_tab:
			audio_input = gr.Files(label=i18n("上传一个或多个音频文件"), type="filepath")
	with gr.Row():
		store_dir = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=False, scale=4)
	with gr.Accordion(i18n("[点击展开] 推理参数设置, 不同模型之间参数相互独立"), open=False):
		gr.Markdown(
			value=i18n(
				"只有在点击保存后才会生效。参数直接写入配置文件, 无法撤销。假如不知道如何设置, 请保持默认值。<br>请牢记自己修改前的参数数值, 防止出现问题以后无法恢复。请确保输入正确的参数, 否则可能会导致模型无法正常运行。<br>假如修改后无法恢复, 请点击``重置``按钮, 这会使得配置文件恢复到默认值。"
			)
		)
		with gr.Row():
			batch_size = gr.Slider(label="batch_size", info=i18n("批次大小, 减小此值可以降低显存占用, 此参数对推理效果影响不大"), value=None, minimum=1, maximum=16, step=1)
			num_overlap = gr.Slider(label="overlap", info=i18n("重叠数, 增大此值可以提高分离效果, 但会增加处理时间, 建议设置成4"), value=None, minimum=1, maximum=16, step=1)
			chunk_size = gr.Slider(label="chunk_size", info=i18n("分块大小, 增大此值可以提高分离效果, 但会增加处理时间和显存占用"), value=None, minimum=44100, maximum=1323000, step=22050)
		with gr.Row():
			normalize = gr.Checkbox(label="normalize", info=i18n("音频归一化, 对音频进行归一化输入和输出, 部分模型没有此功能"), value=False, interactive=False)
			use_tta = gr.Checkbox(label="use_tta", info=i18n("启用TTA, 能小幅提高分离质量, 若使用, 推理时间x3"), value=False, interactive=True)
		with gr.Row():
			save_config_button = gr.Button(i18n("保存配置"))
			reset_config_button = gr.Button(i18n("重置配置"))
	inference_audio = gr.Button(i18n("输入音频分离"), variant="primary", visible=False)
	inference_folder = gr.Button(i18n("输入文件夹分离"), variant="primary", visible=True)
	with gr.Row():
		output_message = gr.Textbox(label="Output Message", scale=5)
		stop_msst = gr.Button(i18n("强制停止"), scale=1)

	audio_tab.select(fn=change_to_audio_infer, outputs=[inference_audio, inference_folder])
	folder_tab.select(fn=change_to_folder_infer, outputs=[inference_audio, inference_folder])
	inference_audio.click(fn=cloud_msst_infer_audio, inputs=[selected_model, audio_input, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta], outputs=output_message)
	inference_folder.click(fn=cloud_msst_infer_folder, inputs=[selected_model, folder_input, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta], outputs=output_message)
	selected_model.change(fn=update_inference_settings, inputs=selected_model, outputs=[batch_size, num_overlap, chunk_size, normalize, extract_instrumental])
	save_config_button.click(fn=save_model_config, inputs=[selected_model, batch_size, num_overlap, chunk_size, normalize], outputs=output_message)
	select_model_type.change(fn=load_msst_cloud_model, inputs=select_model_type, outputs=selected_model)
	reset_config_button.click(fn=reset_model_config, inputs=selected_model, outputs=output_message)
	stop_msst.click(fn=stop_msst_inference)


def vr():
	from webui.vr import stop_vr_inference, load_vr_model_stem

	gr.Markdown(
		value=i18n(
			"说明: 本整合包仅融合了UVR的VR Architecture模型, MDX23C和HtDemucs类模型可以直接使用前面的MSST音频分离。<br>UVR分离使用项目: [https://github.com/nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 并进行了优化。"
		)
	)
	vr_select_model = gr.Dropdown(
		label=i18n("选择模型"), choices=load_vr_cloud_model(), value=webui_config["inference"]["vr_select_model"] if webui_config["inference"]["vr_select_model"] else None, interactive=True
	)
	with gr.Row():
		vr_window_size = gr.Dropdown(label=i18n("Window Size: 窗口大小, 用于平衡速度和质量"), choices=[320, 512, 1024], value="512", interactive=True, allow_custom_value=True)
		vr_aggression = gr.Number(label=i18n("Aggression: 主干提取强度, 范围-100-100, 人声请选5"), minimum=-100, maximum=100, value=5, interactive=True)
		vr_output_format = gr.Radio(label=i18n("输出格式"), choices=["wav", "flac", "mp3"], value="wav", interactive=True)
	with gr.Row():
		vr_use_cpu = gr.Checkbox(label=i18n("使用CPU"), value=force_cpu_value, interactive=False if force_cpu_value else True)
		vr_primary_stem_only = gr.Checkbox(label="primary_only", value=False, interactive=True)
		vr_secondary_stem_only = gr.Checkbox(label="secondary_only", value=False, interactive=True)
	with gr.Tabs():
		with gr.TabItem(label=i18n("输入文件夹")) as folder_tab:
			folder_input = gr.Textbox(label=i18n("输入目录"), value="input/", interactive=False, scale=4)
		with gr.TabItem(label=i18n("输入音频")) as audio_tab:
			audio_input = gr.Files(label="上传一个或多个音频文件", type="filepath")
	with gr.Row():
		vr_store_dir = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=False, scale=4)
	with gr.Accordion(i18n("[点击展开] 以下是一些高级设置, 一般保持默认即可"), open=False):
		with gr.Row():
			vr_batch_size = gr.Slider(label="Batch Size", info=i18n("批次大小, 减小此值可以降低显存占用"), minimum=1, maximum=32, step=1, value=2, interactive=True)
			vr_post_process_threshold = gr.Slider(label="Post Process Threshold", info=i18n("后处理特征阈值, 取值为0.1-0.3, 默认0.2"), minimum=0.1, maximum=0.3, step=0.01, value=0.2, interactive=True)
		with gr.Row():
			vr_enable_tta = gr.Checkbox(label="Enable TTA", info=i18n("启用“测试时增强”, 可能会提高质量, 但速度稍慢"), value=False, interactive=True)
			vr_high_end_process = gr.Checkbox(label="High End Process", info=i18n("将输出音频缺失的频率范围镜像输出, 作用不大"), value=False, interactive=True)
			vr_enable_post_process = gr.Checkbox(label="Enable Post Process", info=i18n("识别人声输出中残留的人工痕迹, 可改善某些歌曲的分离效果"), value=False, interactive=True)
	vr_inference_audio = gr.Button(i18n("输入音频分离"), variant="primary", visible=False)
	vr_inference_folder = gr.Button(i18n("输入文件夹分离"), variant="primary", visible=True)
	with gr.Row():
		vr_output_message = gr.Textbox(label="Output Message", scale=5)
		stop_vr = gr.Button(i18n("强制停止"), scale=1)

	audio_tab.select(fn=change_to_audio_infer, outputs=[vr_inference_audio, vr_inference_folder])
	folder_tab.select(fn=change_to_folder_infer, outputs=[vr_inference_audio, vr_inference_folder])
	vr_select_model.change(fn=load_vr_model_stem, inputs=vr_select_model, outputs=[vr_primary_stem_only, vr_secondary_stem_only])
	vr_inference_audio.click(
		fn=cloud_vr_infer_audio,
		inputs=[
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
		],
		outputs=vr_output_message,
	)
	vr_inference_folder.click(
		fn=cloud_vr_infer_folder,
		inputs=[
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
		],
		outputs=vr_output_message,
	)
	stop_vr.click(fn=stop_vr_inference)


def preset():
	from webui.preset import (
		get_presets_list,
		update_model_stem,
		add_to_flow_func,
		save_flow_func,
		reset_flow_func,
		reset_last_func,
		delete_func,
		load_preset,
		restore_preset_func,
		preset_backup_list,
		stop_preset,
	)

	gr.Markdown(value=i18n("预设流程允许按照预设的顺序运行多个模型。每一个模型的输出将作为下一个模型的输入。"))
	with gr.Tabs():
		with gr.TabItem(label=i18n("使用预设")):
			gr.Markdown(value=i18n("该模式下的UVR推理参数将直接沿用UVR分离页面的推理参数, 如需修改请前往UVR分离页面。<br>修改完成后, 还需要任意处理一首歌才能保存参数! "))
			with gr.Row():
				preset_dropdown = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list(), value=None, interactive=True, scale=4)
				output_format_flow = gr.Radio(label=i18n("输出格式"), choices=["wav", "mp3", "flac"], value="wav", interactive=True, scale=1)
			with gr.Row():
				force_cpu = gr.Checkbox(label=i18n("使用CPU (注意: 使用CPU会导致速度非常慢) "), value=force_cpu_value, interactive=False if force_cpu_value else True)
				use_tta = gr.Checkbox(label=i18n("使用TTA (测试时增强), 可能会提高质量, 但时间x3"), value=False, interactive=True)
				extra_output_dir = gr.Checkbox(label=i18n("将次级输出保存至输出目录的单独文件夹内"), value=False, interactive=True)
			with gr.Tabs():
				with gr.TabItem(label=i18n("输入文件夹")) as folder_tab:
					input_folder = gr.Textbox(label=i18n("输入目录"), value="input/", interactive=False, scale=3)
				with gr.TabItem(label=i18n("输入音频")) as audio_tab:
					input_audio = gr.Files(label=i18n("上传一个或多个音频文件"), type="filepath")
			with gr.Row():
				store_dir_flow = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=False, scale=4)
			inference_audio = gr.Button(i18n("输入音频分离"), variant="primary", visible=False)
			inference_folder = gr.Button(i18n("输入文件夹分离"), variant="primary", visible=True)
			with gr.Row():
				output_message_flow = gr.Textbox(label="Output Message", scale=5)
				stop_preset_inference = gr.Button(i18n("强制停止"), scale=1)
		with gr.TabItem(label=i18n("制作预设")):
			preset_name_input = gr.Textbox(label=i18n("预设名称"), placeholder=i18n("请输入预设名称"), interactive=True)
			with gr.Row():
				model_type = gr.Dropdown(label=i18n("选择模型类型"), choices=MODEL_CHOICES, interactive=True)
				model_name = gr.Dropdown(label=i18n("选择模型"), choices=None, interactive=False, scale=3)
			with gr.Row():
				input_to_next = gr.Radio(label=i18n("作为下一模型输入(或结果输出)的音轨"), choices=[i18n("请先选择模型")], interactive=False)
				output_to_storage = gr.CheckboxGroup(label=i18n("直接保存至输出目录的音轨(可多选)"), choices=[i18n("请先选择模型")], interactive=False)
			add_to_flow = gr.Button(i18n("添加至流程"))
			gr.Markdown(i18n("预设流程"))
			preset_flow = gr.Dataframe(pd.DataFrame({"model_type": [""], "model_name": [""], "input_to_next": [""], "output_to_storage": [""]}), interactive=False, label=None)
			with gr.Row():
				reset_last = gr.Button(i18n("撤销上一步"))
				reset_flow = gr.Button(i18n("全部清空"))
			save_flow = gr.Button(i18n("保存上述预设流程"), variant="primary")
			output_message_make = gr.Textbox(label="Output Message")
		with gr.TabItem(label=i18n("管理预设")):
			gr.Markdown(
				i18n(
					"此页面提供查看预设, 删除预设, 备份预设, 恢复预设等功能<br>`model_type`: 模型类型；`model_name`: 模型名称；`input_to_next`: 作为下一模型输入的音轨；`output_to_storage`: 直接保存至输出目录下的direct_output文件夹内的音轨, **不会经过后续流程处理**<br>每次点击删除预设按钮时, 将自动备份预设以免误操作。"
				)
			)
			with gr.Row():
				preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=get_presets_list(), interactive=True, scale=5)
				delete_button = gr.Button(i18n("删除所选预设"))
			preset_flow_delete = gr.Dataframe(
				pd.DataFrame({"model_type": [i18n("请先选择预设")], "model_name": [i18n("请先选择预设")], "input_to_next": [i18n("请先选择预设")], "output_to_storage": [i18n("请先选择预设")]}),
				interactive=False,
				label=None,
			)
			with gr.Row():
				select_preset_backup = gr.Dropdown(label=i18n("选择需要恢复的预设流程备份"), choices=preset_backup_list(), interactive=True, scale=4)
				restore_preset = gr.Button(i18n("恢复所选预设"), scale=1)
			output_message_manage = gr.Textbox(label="Output Message")

	audio_tab.select(fn=change_to_audio_infer, outputs=[inference_audio, inference_folder])
	folder_tab.select(fn=change_to_folder_infer, outputs=[inference_audio, inference_folder])
	inference_folder.click(fn=cloud_preset_infer_folder, inputs=[input_folder, store_dir_flow, preset_dropdown, force_cpu, output_format_flow, use_tta, extra_output_dir], outputs=output_message_flow)
	inference_audio.click(fn=cloud_preset_infer_audio, inputs=[input_audio, store_dir_flow, preset_dropdown, force_cpu, output_format_flow, use_tta, extra_output_dir], outputs=output_message_flow)
	model_name.change(fn=update_model_stem, inputs=[model_type, model_name], outputs=[input_to_next, output_to_storage])
	add_to_flow.click(fn=add_to_flow_func, inputs=[model_type, model_name, input_to_next, output_to_storage, preset_flow], outputs=preset_flow)
	save_flow.click(fn=save_flow_func, inputs=[preset_name_input, preset_flow], outputs=[output_message_make, preset_name_delete, preset_dropdown])
	delete_button.click(fn=delete_func, inputs=preset_name_delete, outputs=[output_message_manage, preset_name_delete, preset_dropdown, preset_flow_delete, select_preset_backup])
	restore_preset.click(fn=restore_preset_func, inputs=select_preset_backup, outputs=[output_message_manage, preset_dropdown, preset_name_delete, preset_flow_delete])
	model_type.change(load_preset_cloud_model, inputs=model_type, outputs=model_name)
	preset_name_delete.change(load_preset, inputs=preset_name_delete, outputs=preset_flow_delete)
	reset_flow.click(reset_flow_func, outputs=preset_flow)
	reset_last.click(reset_last_func, inputs=preset_flow, outputs=preset_flow)
	stop_preset_inference.click(stop_preset)


def ensemble():
	from webui.ensemble import ensemble_files, update_model_stem, add_to_ensemble_flow, reset_flow_func, reset_last_func, save_ensemble_preset_func, load_ensemble, stop_ensemble_func

	if webui_config["inference"]["ensemble_preset"]:
		accordion_open = False
	else:
		accordion_open = True

	gr.Markdown(
		value=i18n(
			"合奏模式可用于集成不同算法的结果, 具体的文档位于/docs/ensemble.md。目前主要有以下两种合奏方式:<br>1. 从原始音频合奏: 直接上传一个或多个音频文件, 然后选择多个模型进行处理, 将这些处理结果根据选择的合奏模式进行合奏<br>2. 从分离结果合奏: 上传多个已经分离完成的结果音频, 然后选择合奏模式进行合奏"
		)
	)
	with gr.Tabs():
		with gr.TabItem(i18n("从原始音频合奏")):
			gr.Markdown(
				value=i18n(
					"从原始音频合奏需要上传至少一个音频文件, 然后选择多个模型先进行分离处理, 然后将这些处理结果根据选择的合奏模式进行合奏。<br>注意, 请确保你的磁盘空间充足, 合奏过程会产生的临时文件仅会在处理结束后删除。"
				)
			)
			with gr.Accordion(label=i18n("制作合奏流程"), open=accordion_open):
				with gr.Row():
					model_type = gr.Dropdown(label=i18n("选择模型类型"), choices=MODEL_CHOICES, interactive=True)
					model_name = gr.Dropdown(label=i18n("选择模型"), choices=None, interactive=False, scale=3)
				with gr.Row():
					stem_weight = gr.Number(label=i18n("权重"), value=1.0, minimum=0.0, step=0.1, interactive=True)
					stem = gr.Radio(label=i18n("输出音轨"), choices=[i18n("请先选择模型")], interactive=False, scale=3)
				add_to_flow = gr.Button(i18n("添加到合奏流程"), variant="primary")
				with gr.Row():
					reset_last = gr.Button(i18n("撤销上一步"))
					reset_flow = gr.Button(i18n("全部清空"))
				gr.Markdown(i18n("合奏流程"))
				ensemble_flow = gr.Dataframe(value=load_ensemble(), interactive=False, label=None)
				save_ensemble_preset = gr.Button(i18n("保存此合奏流程"), variant="primary")
			with gr.Row():
				ensemble_model_mode = gr.Radio(choices=ENSEMBLE_MODES, label=i18n("集成模式"), value="avg_wave", interactive=True, scale=3)
				output_format = gr.Radio(label=i18n("输出格式"), choices=["wav", "flac", "mp3"], value="wav", interactive=True, scale=1)
			with gr.Row():
				force_cpu = gr.Checkbox(label=i18n("使用CPU (注意: 使用CPU会导致速度非常慢) "), value=force_cpu_value, interactive=True)
				use_tta = gr.Checkbox(label=i18n("使用TTA (测试时增强), 可能会提高质量, 但时间x3"), value=False, interactive=True)
				extract_inst = gr.Checkbox(label=i18n("输出次级音轨 (例如: 合奏人声时, 同时输出伴奏)"), value=False, interactive=True)
			with gr.Tabs():
				with gr.TabItem(label=i18n("输入文件夹")) as folder_tab:
					input_folder = gr.Textbox(label=i18n("输入目录"), value="input/", interactive=False, scale=3)
				with gr.TabItem(label=i18n("输入音频")) as audio_tab:
					input_audio = gr.Files(label=i18n("上传一个或多个音频文件"), type="filepath")
			with gr.Row():
				store_dir_flow = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=False, scale=4)
			inference_audio = gr.Button(i18n("输入音频分离"), variant="primary", visible=False)
			inference_folder = gr.Button(i18n("输入文件夹分离"), variant="primary", visible=True)
		with gr.TabItem(i18n("从分离结果合奏")):
			gr.Markdown(value=i18n("从分离结果合奏需要上传至少两个音频文件, 这些音频文件是使用不同的模型分离同一段音频的结果。因此, 上传的所有音频长度应该相同。"))
			with gr.Row():
				files = gr.Files(label=i18n("上传多个音频文件"), type="filepath", file_count="multiple")
				with gr.Column():
					weights = gr.Textbox(label=i18n("权重(以空格分隔, 数量要与上传的音频一致)"), value="1 1")
					ensembl_output_path = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=False)
			with gr.Row():
				ensemble_type = gr.Radio(choices=ENSEMBLE_MODES, label=i18n("集成模式"), value="avg_wave", interactive=True, scale=3)
				file_output_format = gr.Radio(label=i18n("输出格式"), choices=["wav", "flac", "mp3"], value="wav", interactive=True)
			ensemble_button = gr.Button(i18n("运行"), variant="primary")

	with gr.Row():
		output_message_ensemble = gr.Textbox(label="Output Message", scale=5)
		stop_ensemble = gr.Button(i18n("强制停止"), scale=1)
	with gr.Row():
		with gr.Column():
			gr.Markdown(i18n("### 集成模式"))
			gr.Markdown(
				i18n(
					"1. `avg_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的平均值<br>2. `median_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的中位数<br>3. `min_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的最小绝对值<br>4. `max_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的最大绝对值<br>5. `avg_fft`: 在频谱图 (短时傅里叶变换 (STFT) 2D变体) 上进行集成, 独立地找到频谱图的每个像素的平均值。平均后使用逆STFT得到原始的1D波形<br>6. `median_fft`: 与avg_fft相同, 但使用中位数代替平均值 (仅在集成3个或更多来源时有用) <br>7. `min_fft`: 与avg_fft相同, 但使用最小函数代替平均值 (减少激进程度) <br>8. `max_fft`: 与avg_fft相同, 但使用最大函数代替平均值 (增加激进程度) "
				)
			)
		with gr.Column():
			gr.Markdown(i18n("### 注意事项"))
			gr.Markdown(
				i18n(
					"1. min_fft可用于进行更保守的合成, 它将减少更激进模型的影响。<br>2. 最好合成等质量的模型。在这种情况下, 它将带来增益。如果其中一个模型质量不好, 它将降低整体质量。<br>3. 在原仓库作者的实验中, 与其他方法相比, avg_wave在SDR分数上总是更好或相等。<br>4. 最终会在输出目录下生成一个`ensemble_<集成模式>.wav`。"
				)
			)

	audio_tab.select(fn=change_to_audio_infer, outputs=[inference_audio, inference_folder])
	folder_tab.select(fn=change_to_folder_infer, outputs=[inference_audio, inference_folder])
	add_to_flow.click(fn=add_to_ensemble_flow, inputs=[model_type, model_name, stem_weight, stem, ensemble_flow], outputs=ensemble_flow)
	inference_audio.click(fn=cloud_ensemble_infer_audio, inputs=[ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_audio, extract_inst], outputs=output_message_ensemble)
	inference_folder.click(fn=cloud_ensemble_infer_folder, inputs=[ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_folder, extract_inst], outputs=output_message_ensemble)
	ensemble_button.click(fn=ensemble_files, inputs=[files, ensemble_type, weights, ensembl_output_path, file_output_format], outputs=output_message_ensemble)
	model_type.change(load_preset_cloud_model, inputs=model_type, outputs=model_name)
	model_name.change(fn=update_model_stem, inputs=[model_type, model_name], outputs=[stem])
	reset_flow.click(reset_flow_func, outputs=ensemble_flow)
	reset_last.click(reset_last_func, inputs=ensemble_flow, outputs=ensemble_flow)
	save_ensemble_preset.click(save_ensemble_preset_func, inputs=ensemble_flow)
	stop_ensemble.click(stop_ensemble_func)


def train():
	from webui.train import save_training_config, start_training, update_train_start_check_point, validate_model, stop_msst_valid

	gr.Markdown(
		value=i18n(
			"此页面提供数据集制作教程, 训练参数选择, 以及一键训练。有关配置文件的修改和数据集文件夹的详细说明请参考MSST原项目: [https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)<br>在开始下方的模型训练之前, 请先进行训练数据的制作。<br>说明: 数据集类型即训练集制作Step 1中你选择的类型, 1: Type1; 2: Type2; 3: Type3; 4: Type4, 必须与你的数据集类型相匹配。"
		)
	)
	with gr.Tabs():
		with gr.TabItem(label=i18n("训练")):
			with gr.Row():
				train_model_type = gr.Dropdown(
					label=i18n("选择训练模型类型"), choices=MODEL_TYPE, value=webui_config["training"]["model_type"] if webui_config["training"]["model_type"] else None, interactive=True, scale=1
				)
				train_config_path = gr.Textbox(
					label=i18n("配置文件路径"),
					value=webui_config["training"]["config_path"] if webui_config["training"]["config_path"] else i18n("请输入配置文件路径或选择配置文件"),
					interactive=True,
					scale=3,
				)
			with gr.Row():
				train_dataset_type = gr.Radio(
					label=i18n("数据集类型"), choices=[1, 2, 3, 4], value=webui_config["training"]["dataset_type"] if webui_config["training"]["dataset_type"] else None, interactive=True, scale=1
				)
				train_dataset_path = gr.Textbox(
					label=i18n("数据集路径"),
					value=webui_config["training"]["dataset_path"] if webui_config["training"]["dataset_path"] else i18n("请输入或选择数据集文件夹"),
					interactive=True,
					scale=3,
				)
			with gr.Row():
				train_valid_path = gr.Textbox(
					label=i18n("验证集路径"), value=webui_config["training"]["valid_path"] if webui_config["training"]["valid_path"] else i18n("请输入或选择验证集文件夹"), interactive=True, scale=4
				)
			with gr.Row():
				train_results_path = gr.Textbox(
					label=i18n("模型保存路径"),
					value=webui_config["training"]["results_path"] if webui_config["training"]["results_path"] else i18n("请输入或选择模型保存文件夹"),
					interactive=True,
					scale=3,
				)
			with gr.Row():
				train_start_check_point = gr.Dropdown(label=i18n("初始模型: 继续训练或微调模型训练时, 请选择初始模型, 否则将从头开始训练! "), choices=["None"], value="None", interactive=True, scale=4)
				reflesh_start_check_point = gr.Button(i18n("刷新初始模型列表"), scale=1)
			with gr.Accordion(i18n("训练参数设置"), open=True):
				with gr.Row():
					train_device_ids = gr.CheckboxGroup(
						label=i18n("选择使用的GPU"), choices=device, value=webui_config["training"]["device"] if webui_config["training"]["device"] else device[0], interactive=True
					)
					train_num_workers = gr.Number(
						label=i18n("num_workers: 数据集读取线程数, 0为自动"),
						value=webui_config["training"]["num_workers"] if webui_config["training"]["num_workers"] else 0,
						interactive=True,
						minimum=0,
						maximum=cpu_count(),
						step=1,
					)
					train_seed = gr.Number(label=i18n("随机数种子, 0为随机"), value=0)
				with gr.Row():
					train_pin_memory = gr.Checkbox(label=i18n("是否将加载的数据放置在固定内存中, 默认为否"), value=webui_config["training"]["pin_memory"], interactive=True)
					train_accelerate = gr.Checkbox(label=i18n("是否使用加速训练, 对于多显卡用户会加快训练"), value=webui_config["training"]["accelerate"], interactive=True)
					train_pre_validate = gr.Checkbox(label=i18n("是否在训练前验证模型, 默认为否"), value=webui_config["training"]["pre_valid"], interactive=True)
				with gr.Row():
					train_use_multistft_loss = gr.Checkbox(label=i18n("是否使用MultiSTFT Loss, 默认为否"), value=webui_config["training"]["use_multistft_loss"], interactive=True)
					train_use_mse_loss = gr.Checkbox(label=i18n("是否使用MSE loss, 默认为否"), value=webui_config["training"]["use_mse_loss"], interactive=True)
					train_use_l1_loss = gr.Checkbox(label=i18n("是否使用L1 loss, 默认为否"), value=webui_config["training"]["use_l1_loss"], interactive=True)
				with gr.Row():
					train_metrics_list = gr.CheckboxGroup(
						label=i18n("选择输出的评估指标"), choices=METRICS, value=webui_config["training"]["metrics"] if webui_config["training"]["metrics"] else METRICS[0], interactive=True
					)
					train_metrics_scheduler = gr.Radio(
						label=i18n("选择调度器使用的评估指标"),
						choices=METRICS,
						value=webui_config["training"]["metrics_scheduler"] if webui_config["training"]["metrics_scheduler"] else METRICS[0],
						interactive=True,
					)
			save_train_config = gr.Button(i18n("保存上述训练配置"))
			start_train_button = gr.Button(i18n("开始训练"), variant="primary")
			gr.Markdown(value=i18n("点击开始训练后, 请到终端查看训练进度或报错, 下方不会输出报错信息, 想要停止训练可以直接关闭终端。在训练过程中, 你也可以关闭网页, 仅**保留终端**。"))
			output_message_train = gr.Textbox(label="Output Message", scale=4)

			reflesh_start_check_point.click(fn=update_train_start_check_point, inputs=train_results_path, outputs=train_start_check_point)
			save_train_config.click(
				fn=save_training_config,
				inputs=[
					train_model_type,
					train_config_path,
					train_dataset_type,
					train_dataset_path,
					train_valid_path,
					train_num_workers,
					train_device_ids,
					train_seed,
					train_pin_memory,
					train_use_multistft_loss,
					train_use_mse_loss,
					train_use_l1_loss,
					train_results_path,
					train_accelerate,
					train_pre_validate,
					train_metrics_list,
					train_metrics_scheduler,
				],
				outputs=output_message_train,
			)
			start_train_button.click(
				fn=start_training,
				inputs=[
					train_model_type,
					train_config_path,
					train_dataset_type,
					train_dataset_path,
					train_valid_path,
					train_num_workers,
					train_device_ids,
					train_seed,
					train_pin_memory,
					train_use_multistft_loss,
					train_use_mse_loss,
					train_use_l1_loss,
					train_results_path,
					train_start_check_point,
					train_accelerate,
					train_pre_validate,
					train_metrics_list,
					train_metrics_scheduler,
				],
				outputs=output_message_train,
			)

		with gr.TabItem(label=i18n("验证")):
			gr.Markdown(
				value=i18n(
					"此页面用于手动验证模型效果, 测试验证集, 输出SDR测试信息。输出的信息会存放在输出文件夹的results.txt中。<br>下方参数将自动加载训练页面的参数, 在训练页面点击保存训练参数后, 重启WebUI即可自动加载。当然你也可以手动输入参数。<br>"
				)
			)
			with gr.Row():
				valid_model_type = gr.Dropdown(
					label=i18n("选择模型类型"), choices=MODEL_TYPE, value=webui_config["training"]["model_type"] if webui_config["training"]["model_type"] else None, interactive=True, scale=1
				)
				valid_config_path = gr.Textbox(
					label=i18n("配置文件路径"),
					value=webui_config["training"]["config_path"] if webui_config["training"]["config_path"] else i18n("请输入配置文件路径或选择配置文件"),
					interactive=True,
					scale=3,
				)
			with gr.Row():
				valid_model_path = gr.Textbox(label=i18n("模型路径"), value=i18n("请输入或选择模型文件"), interactive=True, scale=4)
			with gr.Row():
				valid_path = gr.Textbox(
					label=i18n("验证集路径"), value=webui_config["training"]["valid_path"] if webui_config["training"]["valid_path"] else i18n("请输入或选择验证集文件夹"), interactive=True, scale=4
				)
			with gr.Row():
				valid_results_path = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=True, scale=3)
			with gr.Accordion(i18n("验证参数设置"), open=True):
				with gr.Row():
					valid_device_ids = gr.CheckboxGroup(
						label=i18n("选择使用的GPU"), choices=device, value=webui_config["training"]["device"] if webui_config["training"]["device"] else device[0], interactive=True
					)
					valid_extension = gr.Radio(label=i18n("选择验证集音频格式"), choices=["wav", "flac", "mp3"], value="wav", interactive=True)
					valid_num_workers = gr.Number(
						label=i18n("验证集读取线程数, 0为自动"),
						value=webui_config["training"]["num_workers"] if webui_config["training"]["num_workers"] else 0,
						interactive=True,
						minimum=0,
						maximum=cpu_count(),
						step=1,
					)

				with gr.Row():
					with gr.Column():
						vaild_metrics = gr.CheckboxGroup(
							label=i18n("选择输出的评估指标"), choices=METRICS, value=webui_config["training"]["metrics"] if webui_config["training"]["metrics"] else METRICS[0], interactive=True
						)
					with gr.Column():
						valid_pin_memory = gr.Checkbox(label=i18n("是否将加载的数据放置在固定内存中, 默认为否"), value=webui_config["training"]["pin_memory"], interactive=True)
						valid_use_tta = gr.Checkbox(label=i18n("启用TTA, 能小幅提高分离质量, 若使用, 推理时间x3"), value=False, interactive=True)
			valid_button = gr.Button(i18n("开始验证"), variant="primary")
			with gr.Row():
				valid_output_message = gr.Textbox(label="Output Message", scale=4)
				stop_valid = gr.Button(i18n("强制停止"), scale=1)

			valid_button.click(
				fn=validate_model,
				inputs=[
					valid_model_type,
					valid_config_path,
					valid_model_path,
					valid_path,
					valid_results_path,
					valid_device_ids,
					valid_num_workers,
					valid_extension,
					valid_pin_memory,
					valid_use_tta,
					vaild_metrics,
				],
				outputs=valid_output_message,
			)
			stop_valid.click(fn=stop_msst_valid)


def tools():
	from webui.tools import convert_audio, merge_audios, caculate_sdr, some_inference

	with gr.Tabs():
		with gr.TabItem(label=i18n("音频格式转换")):
			gr.Markdown(
				value=i18n(
					"上传一个或多个音频文件并将其转换为指定格式。<br>支持的格式包括 .mp3, .flac, .wav, .ogg, .m4a, .wma, .aac...等等。<br>**不支持**网易云音乐/QQ音乐等加密格式, 如.ncm, .qmc等。"
				)
			)
			with gr.Row():
				inputs = gr.Files(label=i18n("上传一个或多个音频文件"))
				with gr.Column():
					ffmpeg_output_format = gr.Dropdown(
						label=i18n("选择或输入音频输出格式"), choices=["wav", "flac", "mp3", "ogg", "m4a", "wma", "aac"], value="wav", allow_custom_value=True, interactive=True
					)
					ffmpeg_output_folder = gr.Textbox(label=i18n("选择音频输出目录"), value="results/", interactive=False)
			with gr.Row():
				sample_rate = gr.Radio(label=i18n("输出音频采样率(Hz)"), choices=[32000, 44100, 48000], value=44100, interactive=True)
				channels = gr.Radio(label=i18n("输出音频声道数"), choices=[i18n("单声道"), i18n("立体声")], value=i18n("立体声"), interactive=True)
			with gr.Row():
				wav_bit_depth = gr.Radio(label=i18n("输出wav位深度"), choices=["PCM-16", "PCM-24", "PCM-32"], value="PCM-16", interactive=True)
				flac_bit_depth = gr.Radio(label=i18n("输出flac位深度"), choices=["16-bit", "32-bit"], value="16-bit", interactive=True)
				mp3_bit_rate = gr.Radio(label=i18n("输出mp3比特率(bps)"), choices=["192k", "256k", "320k"], value="320k", interactive=True)
				ogg_bit_rate = gr.Radio(label=i18n("输出ogg比特率(bps)"), choices=["192k", "320k", "450k"], value="320k", interactive=True)
			convert_audio_button = gr.Button(i18n("转换音频"), variant="primary")
			output_message_ffmpeg = gr.Textbox(label="Output Message")
		with gr.TabItem(label=i18n("合并音频")):
			gr.Markdown(
				value=i18n(
					"点击合并音频按钮后, 将自动把输入文件夹中的所有音频文件合并为一整个音频文件<br>目前支持的格式包括 .mp3, .flac, .wav, .ogg, m4a 这五种<br>合并后的音频会保存至输出目录中, 文件名为merged_audio_<文件夹名字>.wav"
				)
			)
			with gr.Row():
				merge_audio_input = gr.Textbox(label=i18n("输入目录"), value="input/", interactive=False, scale=4)
			with gr.Row():
				merge_audio_output = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=False, scale=4)
			merge_audio_button = gr.Button(i18n("合并音频"), variant="primary")
			output_message_merge = gr.Textbox(label="Output Message")
		with gr.TabItem(label=i18n("计算SDR")):
			with gr.Column():
				gr.Markdown(
					value=i18n(
						"上传两个**wav音频文件**并计算它们的[SDR](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021#evaluation-metric)。<br>SDR是一个用于评估模型质量的数值。数值越大, 模型算法结果越好。"
					)
				)
			with gr.Row():
				reference_audio = gr.File(label=i18n("参考音频"), type="filepath")
				estimated_audio = gr.File(label=i18n("待估音频"), type="filepath")
			compute_sdr_button = gr.Button(i18n("计算SDR"), variant="primary")
			output_message_sdr = gr.Textbox(label="Output Message")
		with gr.TabItem(label=i18n("歌声转MIDI")):
			gr.Markdown(
				value=i18n(
					"歌声转MIDI功能使用开源项目[SOME](https://github.com/openvpi/SOME/), 可以将分离得到的**干净的歌声**转换成.mid文件。<br>【必须】若想要使用此功能, 请先下载权重文件[model_steps_64000_simplified.ckpt](https://hf-mirror.com/Sucial/MSST-WebUI/resolve/main/SOME_weights/model_steps_64000_simplified.ckpt)并将其放置在程序目录下的`tools/SOME_weights`文件夹内。文件命名不可随意更改!"
				)
			)
			gr.Markdown(
				value=i18n(
					"如果不知道如何测量歌曲BPM, 可以尝试这两个在线测量工具: [bpmdetector](https://bpmdetector.kniffen.dev/) | [key-bpm-finder](https://vocalremover.org/zh/key-bpm-finder), 测量时建议上传原曲或伴奏, 若干声可能导致测量结果不准确。"
				)
			)
			with gr.Row():
				some_input_audio = gr.File(label=i18n("上传音频"), type="filepath")
				with gr.Column():
					audio_bpm = gr.Number(label=i18n("输入音频BPM"), value=120, interactive=True)
					some_output_folder = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=False, scale=3)
			some_button = gr.Button(i18n("开始转换"), variant="primary")
			output_message_some = gr.Textbox(label="Output Message")
			gr.Markdown(i18n("### 注意事项"))
			gr.Markdown(
				i18n(
					"1. 音频BPM (每分钟节拍数) 可以通过MixMeister BPM Analyzer等软件测量获取。<br>2. 为保证MIDI提取质量, 音频文件请采用干净清晰无混响底噪人声。<br>3. 输出MIDI不带歌词信息, 需要用户自行添加歌词。<br>4. 实际使用体验中部分音符会出现断开的现象, 需自行修正。SOME的模型主要面向DiffSinger唱法模型自动标注, 比正常用户在创作中需要的MIDI更加精细, 因而可能导致模型倾向于对音符进行切分。<br>5. 提取的MIDI没有量化/没有对齐节拍/不适配BPM, 需自行到各编辑器中手动调整。"
				)
			)

	convert_audio_button.click(
		fn=convert_audio, inputs=[inputs, ffmpeg_output_format, ffmpeg_output_folder, sample_rate, channels, wav_bit_depth, flac_bit_depth, mp3_bit_rate, ogg_bit_rate], outputs=output_message_ffmpeg
	)
	merge_audio_button.click(fn=merge_audios, inputs=[merge_audio_input, merge_audio_output], outputs=output_message_merge)
	compute_sdr_button.click(fn=caculate_sdr, inputs=[reference_audio, estimated_audio], outputs=output_message_sdr)
	some_button.click(fn=some_inference, inputs=[some_input_audio, audio_bpm, some_output_folder], outputs=output_message_some)


def settings():
	from webui.settings import reset_settings, reset_webui_config, change_language, save_audio_setting_fn

	theme_choices = []
	for i in os.listdir(THEME_FOLDER):
		if i.endswith(".json"):
			theme_choices.append(i)

	language = "Auto"
	for lg in language_dict.keys():
		current_language = webui_config["settings"]["language"]
		if language_dict[lg] == current_language:
			language = lg

	with gr.Tabs():
		with gr.TabItem(i18n("WebUI设置")):
			with gr.Row():
				gr.Textbox(label=i18n("GPU信息"), value=device if len(device) > 1 else device[0], interactive=False)
				gr.Textbox(label=i18n("系统信息"), value=f"System: {platform.system()}, Machine: {platform.machine()}", interactive=False)
			with gr.Row():
				set_language = gr.Dropdown(label=i18n("选择语言"), choices=language_dict.keys(), value=language, interactive=True)
				debug_mode = gr.Checkbox(label=i18n("全局调试模式: 向开发者反馈问题时请开启。(该选项支持热切换)"), value=webui_config["settings"]["debug"], interactive=True)
			with gr.Row():
				reset_all_webui_config = gr.Button(i18n("重置WebUI路径记录"), variant="primary")
				reset_seetings = gr.Button(i18n("重置WebUI设置"), variant="primary")
			setting_output_message = gr.Textbox(label="Output Message")
			restart_webui = gr.Button(i18n("重启WebUI"), variant="primary")
		with gr.TabItem(label=i18n("音频输出设置")):
			gr.Markdown(i18n("此页面支持用户自定义修改MSST/VR推理后输出音频的质量。输出音频的**采样率, 声道数与模型支持的参数有关, 无法更改**。<br>修改完成后点击保存设置即可生效。"))
			wav_bit_depth = gr.Radio(label=i18n("输出wav位深度"), choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"], value="FLOAT", interactive=True)
			flac_bit_depth = gr.Radio(label=i18n("输出flac位深度"), choices=["PCM_16", "PCM_24"], value="PCM_24", interactive=True)
			mp3_bit_rate = gr.Radio(label=i18n("输出mp3比特率(bps)"), choices=["96k", "128k", "192k", "256k", "320k"], value="320k", interactive=True)
			save_audio_setting = gr.Button(i18n("保存设置"), variant="primary")
			audio_setting_output_message = gr.Textbox(label="Output Message")

	restart_webui.click(fn=webui_restart, outputs=setting_output_message)
	reset_seetings.click(fn=reset_settings, outputs=setting_output_message)
	reset_all_webui_config.click(fn=reset_webui_config, outputs=setting_output_message)
	set_language.change(fn=change_language, inputs=set_language, outputs=setting_output_message)
	debug_mode.change(fn=log_level_debug, inputs=debug_mode, outputs=setting_output_message)
	save_audio_setting.click(fn=save_audio_setting_fn, inputs=[wav_bit_depth, flac_bit_depth, mp3_bit_rate], outputs=audio_setting_output_message)
