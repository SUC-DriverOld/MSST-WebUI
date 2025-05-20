__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr

from webui.utils import i18n, load_vr_model, select_folder, open_folder, change_to_audio_infer, change_to_folder_infer
from webui.init import init_selected_vr_model
from webui.vr import vr_inference_single, vr_inference_multi, stop_vr_inference, load_vr_model_stem


def vr(webui_config, force_cpu_flag=False):
	if webui_config["inference"]["force_cpu"] or force_cpu_flag:
		force_cpu_value = True
	else:
		force_cpu_value = False

	primary_label, secondary_label = init_selected_vr_model()

	gr.Markdown(
		value=i18n(
			"说明: 本整合包仅融合了UVR的VR Architecture模型, MDX23C和HtDemucs类模型可以直接使用前面的MSST音频分离。<br>UVR分离使用项目: [https://github.com/nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 并进行了优化。"
		)
	)
	vr_select_model = gr.Dropdown(
		label=i18n("选择模型"), choices=load_vr_model(), value=webui_config["inference"]["vr_select_model"] if webui_config["inference"]["vr_select_model"] else None, interactive=True
	)
	with gr.Row():
		vr_window_size = gr.Dropdown(
			label=i18n("Window Size: 窗口大小, 用于平衡速度和质量, 默认为512"),
			choices=[320, 512, 1024],
			value=webui_config["inference"]["vr_window_size"] if webui_config["inference"]["vr_window_size"] else 512,
			interactive=True,
			allow_custom_value=True,
		)
		vr_aggression = gr.Number(
			label=i18n("Aggression: 主干提取强度, 范围-100-100, 人声请选5"),
			minimum=-100,
			maximum=100,
			value=webui_config["inference"]["vr_aggression"] if webui_config["inference"]["vr_aggression"] else 5,
			interactive=True,
		)
		vr_output_format = gr.Radio(
			label=i18n("输出格式"), choices=["wav", "flac", "mp3"], value=webui_config["inference"]["output_format"] if webui_config["inference"]["output_format"] else "wav", interactive=True
		)
	with gr.Row():
		vr_use_cpu = gr.Checkbox(label=i18n("使用CPU"), value=force_cpu_value, interactive=False if force_cpu_flag else True)
		vr_primary_stem_only = gr.Checkbox(
			label=primary_label, value=webui_config["inference"]["vr_primary_stem_only"] if webui_config["inference"]["vr_primary_stem_only"] else False, interactive=True
		)
		vr_secondary_stem_only = gr.Checkbox(
			label=secondary_label, value=webui_config["inference"]["vr_secondary_stem_only"] if webui_config["inference"]["vr_secondary_stem_only"] else False, interactive=True
		)
	with gr.Tabs():
		with gr.TabItem(label=i18n("输入音频")) as audio_tab:
			audio_input = gr.Files(label="上传一个或多个音频文件", type="filepath")
		with gr.TabItem(label=i18n("输入文件夹")) as folder_tab:
			with gr.Row():
				folder_input = gr.Textbox(label=i18n("输入目录"), value=webui_config["inference"]["input_dir"] if webui_config["inference"]["input_dir"] else "input/", interactive=True, scale=4)
				vr_select_multi_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
				vr_open_multi_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
	with gr.Row():
		vr_store_dir = gr.Textbox(label=i18n("输出目录"), value=webui_config["inference"]["store_dir"] if webui_config["inference"]["store_dir"] else "results/", interactive=True, scale=4)
		vr_select_store_btn = gr.Button(i18n("选择文件夹"), scale=1)
		vr_open_store_btn = gr.Button(i18n("打开文件夹"), scale=1)
	with gr.Accordion(i18n("[点击展开] 以下是一些高级设置, 一般保持默认即可"), open=False):
		with gr.Row():
			vr_batch_size = gr.Slider(
				label="Batch Size",
				info=i18n("批次大小, 减小此值可以降低显存占用"),
				minimum=1,
				maximum=16,
				step=1,
				value=webui_config["inference"]["vr_batch_size"] if webui_config["inference"]["vr_batch_size"] else 2,
				interactive=True,
			)
			vr_post_process_threshold = gr.Slider(
				label="Post Process Threshold",
				info=i18n("后处理特征阈值, 取值为0.1-0.3, 默认0.2"),
				minimum=0.1,
				maximum=0.3,
				step=0.01,
				value=webui_config["inference"]["vr_post_process_threshold"] if webui_config["inference"]["vr_post_process_threshold"] else 0.2,
				interactive=True,
			)
		with gr.Row():
			vr_enable_tta = gr.Checkbox(
				label="Enable TTA",
				info=i18n("启用“测试时增强”, 可能会提高质量, 但速度稍慢"),
				value=webui_config["inference"]["vr_enable_tta"] if webui_config["inference"]["vr_enable_tta"] else False,
				interactive=True,
			)
			vr_high_end_process = gr.Checkbox(
				label="High End Process",
				info=i18n("将输出音频缺失的频率范围镜像输出, 作用不大"),
				value=webui_config["inference"]["vr_high_end_process"] if webui_config["inference"]["vr_high_end_process"] else False,
				interactive=True,
			)
			vr_enable_post_process = gr.Checkbox(
				label="Enable Post Process",
				info=i18n("识别人声输出中残留的人工痕迹, 可改善某些歌曲的分离效果"),
				value=webui_config["inference"]["vr_enable_post_process"] if webui_config["inference"]["vr_enable_post_process"] else False,
				interactive=True,
			)
	vr_inference_audio = gr.Button(i18n("输入音频分离"), variant="primary", visible=True)
	vr_inference_folder = gr.Button(i18n("输入文件夹分离"), variant="primary", visible=False)
	with gr.Row():
		vr_output_message = gr.Textbox(label="Output Message", scale=5)
		stop_vr = gr.Button(i18n("强制停止"), scale=1)

	audio_tab.select(fn=change_to_audio_infer, outputs=[vr_inference_audio, vr_inference_folder])
	folder_tab.select(fn=change_to_folder_infer, outputs=[vr_inference_audio, vr_inference_folder])

	vr_select_model.change(fn=load_vr_model_stem, inputs=vr_select_model, outputs=[vr_primary_stem_only, vr_secondary_stem_only])
	vr_inference_audio.click(
		fn=vr_inference_single,
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
		fn=vr_inference_multi,
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
	vr_select_multi_input_dir.click(fn=select_folder, outputs=folder_input)
	vr_open_multi_input_dir.click(fn=open_folder, inputs=folder_input)
	vr_select_store_btn.click(fn=select_folder, outputs=vr_store_dir)
	vr_open_store_btn.click(fn=open_folder, inputs=vr_store_dir)
	stop_vr.click(fn=stop_vr_inference)
