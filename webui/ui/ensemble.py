__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr

from utils.constant import *
from webui.utils import i18n, select_folder, open_folder, update_model_name, change_to_audio_infer, change_to_folder_infer
from webui.ensemble import (
	ensemble_files,
	update_model_stem,
	add_to_ensemble_flow,
	reset_flow_func,
	reset_last_func,
	save_ensemble_preset_func,
	load_ensemble,
	inference_audio_func,
	inference_folder_func,
	stop_ensemble_func,
)


def ensemble(webui_config, force_cpu_flag=False):
	if webui_config["inference"]["force_cpu"] or force_cpu_flag:
		force_cpu_value = True
	else:
		force_cpu_value = False

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
				ensemble_model_mode = gr.Radio(
					choices=ENSEMBLE_MODES,
					label=i18n("集成模式"),
					value=webui_config["inference"]["ensemble_type"] if webui_config["inference"]["ensemble_type"] else "avg_wave",
					interactive=True,
					scale=3,
				)
				output_format = gr.Radio(
					label=i18n("输出格式"),
					choices=["wav", "flac", "mp3"],
					value=webui_config["inference"]["output_format"] if webui_config["inference"]["output_format"] else "wav",
					interactive=True,
					scale=1,
				)
			with gr.Row():
				force_cpu = gr.Checkbox(label=i18n("使用CPU (注意: 使用CPU会导致速度非常慢) "), value=force_cpu_value, interactive=False if force_cpu_flag else True)
				use_tta = gr.Checkbox(
					label=i18n("使用TTA (测试时增强), 可能会提高质量, 但时间x3"),
					value=webui_config["inference"]["ensemble_use_tta"] if webui_config["inference"]["ensemble_use_tta"] else False,
					interactive=True,
				)
				extract_inst = gr.Checkbox(
					label=i18n("输出次级音轨 (例如: 合奏人声时, 同时输出伴奏)"),
					value=webui_config["inference"]["ensemble_extract_inst"] if webui_config["inference"]["ensemble_extract_inst"] else False,
					interactive=True,
				)
			with gr.Tabs():
				with gr.TabItem(label=i18n("输入音频")) as audio_tab:
					input_audio = gr.Files(label=i18n("上传一个或多个音频文件"), type="filepath")
				with gr.TabItem(label=i18n("输入文件夹")) as folder_tab:
					with gr.Row():
						input_folder = gr.Textbox(
							label=i18n("输入目录"), value=webui_config["inference"]["input_dir"] if webui_config["inference"]["input_dir"] else "input/", interactive=True, scale=3
						)
						select_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
						open_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
			with gr.Row():
				store_dir_flow = gr.Textbox(label=i18n("输出目录"), value=webui_config["inference"]["store_dir"] if webui_config["inference"]["store_dir"] else "results/", interactive=True, scale=4)
				select_output_dir = gr.Button(i18n("选择文件夹"), scale=1)
				open_output_dir = gr.Button(i18n("打开文件夹"), scale=1)
			inference_audio = gr.Button(i18n("输入音频分离"), variant="primary", visible=True)
			inference_folder = gr.Button(i18n("输入文件夹分离"), variant="primary", visible=False)
		with gr.TabItem(i18n("从分离结果合奏")):
			gr.Markdown(value=i18n("从分离结果合奏需要上传至少两个音频文件, 这些音频文件是使用不同的模型分离同一段音频的结果。因此, 上传的所有音频长度应该相同。"))
			with gr.Row():
				files = gr.Files(label=i18n("上传多个音频文件"), type="filepath", file_count="multiple")
				with gr.Column():
					weights = gr.Textbox(label=i18n("权重(以空格分隔, 数量要与上传的音频一致)"), value="1 1")
					ensembl_output_path = gr.Textbox(label=i18n("输出目录"), value=webui_config["inference"]["store_dir"] if webui_config["inference"]["store_dir"] else "results/", interactive=True)
					with gr.Row():
						select_ensembl_output_path = gr.Button(i18n("选择文件夹"))
						open_ensembl_output_path = gr.Button(i18n("打开文件夹"))
			with gr.Row():
				ensemble_type = gr.Radio(
					choices=ENSEMBLE_MODES,
					label=i18n("集成模式"),
					value=webui_config["inference"]["ensemble_type"] if webui_config["inference"]["ensemble_type"] else "avg_wave",
					interactive=True,
					scale=3,
				)
				file_output_format = gr.Radio(
					label=i18n("输出格式"), choices=["wav", "flac", "mp3"], value=webui_config["inference"]["output_format"] if webui_config["inference"]["output_format"] else "wav", interactive=True
				)
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
	inference_audio.click(fn=inference_audio_func, inputs=[ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_audio, extract_inst], outputs=output_message_ensemble)
	inference_folder.click(fn=inference_folder_func, inputs=[ensemble_model_mode, output_format, force_cpu, use_tta, store_dir_flow, input_folder, extract_inst], outputs=output_message_ensemble)
	ensemble_button.click(fn=ensemble_files, inputs=[files, ensemble_type, weights, ensembl_output_path, file_output_format], outputs=output_message_ensemble)
	model_type.change(update_model_name, inputs=model_type, outputs=model_name)
	model_name.change(fn=update_model_stem, inputs=[model_type, model_name], outputs=[stem])
	select_ensembl_output_path.click(fn=select_folder, outputs=ensembl_output_path)
	open_ensembl_output_path.click(fn=open_folder, inputs=ensembl_output_path)
	reset_flow.click(reset_flow_func, outputs=ensemble_flow)
	reset_last.click(reset_last_func, inputs=ensemble_flow, outputs=ensemble_flow)
	save_ensemble_preset.click(save_ensemble_preset_func, inputs=ensemble_flow)
	select_input_dir.click(fn=select_folder, outputs=input_folder)
	open_input_dir.click(fn=open_folder, inputs=input_folder)
	select_output_dir.click(fn=select_folder, outputs=store_dir_flow)
	open_output_dir.click(fn=open_folder, inputs=store_dir_flow)
	stop_ensemble.click(stop_ensemble_func)
