__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr
import pandas as pd

from utils.constant import *
from webui.utils import i18n, select_folder, open_folder, update_model_name, change_to_audio_infer, change_to_folder_infer
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
	preset_inference,
	preset_inference_audio,
	stop_preset,
)


def preset(webui_config, force_cpu_flag=False):
	if webui_config["inference"]["force_cpu"] or force_cpu_flag:
		force_cpu_value = True
	else:
		force_cpu_value = False

	gr.Markdown(value=i18n("预设流程允许按照预设的顺序运行多个模型。每一个模型的输出将作为下一个模型的输入。"))
	with gr.Tabs():
		with gr.TabItem(label=i18n("使用预设")):
			gr.Markdown(value=i18n("该模式下的UVR推理参数将直接沿用UVR分离页面的推理参数, 如需修改请前往UVR分离页面。<br>修改完成后, 还需要任意处理一首歌才能保存参数! "))
			with gr.Row():
				preset_dropdown = gr.Dropdown(
					label=i18n("请选择预设"), choices=get_presets_list(), value=webui_config["inference"]["preset"] if webui_config["inference"]["preset"] else None, interactive=True, scale=4
				)
				output_format_flow = gr.Radio(
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
					value=webui_config["inference"]["preset_use_tta"] if webui_config["inference"]["preset_use_tta"] else False,
					interactive=True,
				)
				extra_output_dir = gr.Checkbox(
					label=i18n("将次级输出保存至输出目录的单独文件夹内"),
					value=webui_config["inference"]["extra_output_dir"] if webui_config["inference"]["extra_output_dir"] else False,
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
				open_preset_backup = gr.Button(i18n("打开备份文件夹"), scale=1)
			output_message_manage = gr.Textbox(label="Output Message")

	audio_tab.select(fn=change_to_audio_infer, outputs=[inference_audio, inference_folder])
	folder_tab.select(fn=change_to_folder_infer, outputs=[inference_audio, inference_folder])

	add_to_flow.click(fn=add_to_flow_func, inputs=[model_type, model_name, input_to_next, output_to_storage, preset_flow], outputs=preset_flow)
	inference_folder.click(fn=preset_inference, inputs=[input_folder, store_dir_flow, preset_dropdown, force_cpu, output_format_flow, use_tta, extra_output_dir], outputs=output_message_flow)
	inference_audio.click(fn=preset_inference_audio, inputs=[input_audio, store_dir_flow, preset_dropdown, force_cpu, output_format_flow, use_tta, extra_output_dir], outputs=output_message_flow)
	model_name.change(fn=update_model_stem, inputs=[model_type, model_name], outputs=[input_to_next, output_to_storage])
	save_flow.click(fn=save_flow_func, inputs=[preset_name_input, preset_flow], outputs=[output_message_make, preset_name_delete, preset_dropdown])
	delete_button.click(fn=delete_func, inputs=preset_name_delete, outputs=[output_message_manage, preset_name_delete, preset_dropdown, preset_flow_delete, select_preset_backup])
	restore_preset.click(fn=restore_preset_func, inputs=select_preset_backup, outputs=[output_message_manage, preset_dropdown, preset_name_delete, preset_flow_delete])
	model_type.change(update_model_name, inputs=model_type, outputs=model_name)
	preset_name_delete.change(load_preset, inputs=preset_name_delete, outputs=preset_flow_delete)
	reset_flow.click(reset_flow_func, outputs=preset_flow)
	select_input_dir.click(fn=select_folder, outputs=input_folder)
	open_input_dir.click(fn=open_folder, inputs=input_folder)
	select_output_dir.click(fn=select_folder, outputs=store_dir_flow)
	open_output_dir.click(fn=open_folder, inputs=store_dir_flow)
	open_preset_backup.click(open_folder, inputs=gr.Textbox(PRESETS_BACKUP, visible=False))
	reset_last.click(reset_last_func, inputs=preset_flow, outputs=preset_flow)
	stop_preset_inference.click(stop_preset)
