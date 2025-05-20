__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr

from webui.utils import i18n, select_folder, open_folder, change_to_audio_infer, change_to_folder_infer
from webui.init import init_selected_model, init_selected_msst_model
from webui.msst import run_inference_single, run_multi_inference, stop_msst_inference, update_selected_model, load_selected_model, save_model_config, reset_model_config, update_inference_settings


def msst(webui_config, device, force_cpu_flag=False):
	device = [value for _, value in device.items()]

	if webui_config["inference"]["force_cpu"] or force_cpu_flag:
		force_cpu_value = True
	else:
		force_cpu_value = False

	batch_size_number, num_overlap_number, chunk_size_number, is_normalize = init_selected_model()

	gr.Markdown(value=i18n("MSST音频分离原项目地址: [https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)"))
	with gr.Row():
		select_model_type = gr.Dropdown(
			label=i18n("选择模型类型"),
			choices=["vocal_models", "multi_stem_models", "single_stem_models"],
			value=webui_config["inference"]["model_type"] if webui_config["inference"]["model_type"] else None,
			interactive=True,
			scale=1,
		)
		selected_model = gr.Dropdown(
			label=i18n("选择模型"), choices=load_selected_model(), value=webui_config["inference"]["selected_model"] if webui_config["inference"]["selected_model"] else None, interactive=True, scale=4
		)
	with gr.Row():
		gpu_id = gr.CheckboxGroup(label=i18n("选择使用的GPU"), choices=device, value=webui_config["inference"]["device"] if webui_config["inference"]["device"] else device[0], interactive=True)
		output_format = gr.Radio(
			label=i18n("输出格式"), choices=["wav", "flac", "mp3"], value=webui_config["inference"]["output_format"] if webui_config["inference"]["output_format"] else "wav", interactive=True
		)
	with gr.Row():
		extract_instrumental = gr.CheckboxGroup(
			label=i18n("选择输出音轨"), choices=init_selected_msst_model(), value=webui_config["inference"]["instrumental"] if webui_config["inference"]["instrumental"] else None, interactive=True
		)
		force_cpu = gr.Checkbox(info=i18n("强制使用CPU推理, 注意: 使用CPU推理速度非常慢!"), label=i18n("使用CPU"), value=force_cpu_value, interactive=False if force_cpu_flag else True)
	with gr.Tabs():
		with gr.TabItem(label=i18n("输入音频")) as audio_tab:
			audio_input = gr.Files(label=i18n("上传一个或多个音频文件"), type="filepath")
		with gr.TabItem(label=i18n("输入文件夹")) as folder_tab:
			with gr.Row():
				folder_input = gr.Textbox(label=i18n("输入目录"), value=webui_config["inference"]["input_dir"] if webui_config["inference"]["input_dir"] else "input/", interactive=True, scale=4)
				select_multi_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
				open_multi_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
	with gr.Row():
		store_dir = gr.Textbox(label=i18n("输出目录"), value=webui_config["inference"]["store_dir"] if webui_config["inference"]["store_dir"] else "results/", interactive=True, scale=4)
		select_store_btn = gr.Button(i18n("选择文件夹"), scale=1)
		open_store_btn = gr.Button(i18n("打开文件夹"), scale=1)
	with gr.Accordion(i18n("[点击展开] 推理参数设置, 不同模型之间参数相互独立"), open=False):
		gr.Markdown(
			value=i18n(
				"只有在点击保存后才会生效。参数直接写入配置文件, 无法撤销。假如不知道如何设置, 请保持默认值。<br>请牢记自己修改前的参数数值, 防止出现问题以后无法恢复。请确保输入正确的参数, 否则可能会导致模型无法正常运行。<br>假如修改后无法恢复, 请点击``重置``按钮, 这会使得配置文件恢复到默认值。"
			)
		)
		with gr.Row():
			batch_size = gr.Slider(label="batch_size", info=i18n("批次大小, 减小此值可以降低显存占用, 此参数对推理效果影响不大"), value=batch_size_number, minimum=1, maximum=16, step=1)
			num_overlap = gr.Slider(label="overlap", info=i18n("重叠数, 增大此值可以提高分离效果, 但会增加处理时间, 建议设置成4"), value=num_overlap_number, minimum=1, maximum=16, step=1)
			chunk_size = gr.Slider(label="chunk_size", info=i18n("分块大小, 增大此值可以提高分离效果, 但会增加处理时间和显存占用"), value=chunk_size_number, minimum=44100, maximum=1323000, step=22050)
		with gr.Row():
			normalize = gr.Checkbox(label="normalize", info=i18n("音频归一化, 对音频进行归一化输入和输出, 部分模型没有此功能"), value=is_normalize, interactive=False)
			use_tta = gr.Checkbox(
				label="use_tta",
				info=i18n("启用TTA, 能小幅提高分离质量, 若使用, 推理时间x3"),
				value=webui_config["inference"]["use_tta"] if webui_config["inference"]["use_tta"] else False,
				interactive=True,
			)
		with gr.Row():
			save_config_button = gr.Button(i18n("保存配置"))
			reset_config_button = gr.Button(i18n("重置配置"))
	inference_audio = gr.Button(i18n("输入音频分离"), variant="primary", visible=True)
	inference_folder = gr.Button(i18n("输入文件夹分离"), variant="primary", visible=False)
	with gr.Row():
		output_message = gr.Textbox(label="Output Message", scale=5)
		stop_msst = gr.Button(i18n("强制停止"), scale=1)

	audio_tab.select(fn=change_to_audio_infer, outputs=[inference_audio, inference_folder])
	folder_tab.select(fn=change_to_folder_infer, outputs=[inference_audio, inference_folder])

	inference_audio.click(fn=run_inference_single, inputs=[selected_model, audio_input, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta], outputs=output_message)
	inference_folder.click(fn=run_multi_inference, inputs=[selected_model, folder_input, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta], outputs=output_message)

	selected_model.change(fn=update_inference_settings, inputs=selected_model, outputs=[batch_size, num_overlap, chunk_size, normalize, extract_instrumental])
	save_config_button.click(fn=save_model_config, inputs=[selected_model, batch_size, num_overlap, chunk_size, normalize], outputs=output_message)
	select_model_type.change(fn=update_selected_model, inputs=select_model_type, outputs=selected_model)
	reset_config_button.click(fn=reset_model_config, inputs=selected_model, outputs=output_message)
	select_store_btn.click(fn=select_folder, outputs=store_dir)
	open_store_btn.click(fn=open_folder, inputs=store_dir)
	select_multi_input_dir.click(fn=select_folder, outputs=folder_input)
	open_multi_input_dir.click(fn=open_folder, inputs=folder_input)
	stop_msst.click(fn=stop_msst_inference)
