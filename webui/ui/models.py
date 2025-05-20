__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr

from utils.constant import *
from webui.utils import i18n, webui_restart
from webui.models import (
	upgrade_download_model_name,
	download_model,
	manual_download_model,
	install_unmsst_model,
	install_unvr_model,
	update_vr_param,
	get_all_model_param,
	open_model_folder,
	open_download_manager,
	show_model_info,
)


def models(webui_config):
	uvr_model_folder = webui_config["settings"]["uvr_model_dir"]

	with gr.Tabs():
		with gr.TabItem(label=i18n("下载官方模型")):
			with gr.Row():
				with gr.Column(scale=3):
					open_downloadmanager = gr.Button(i18n("点击打开下载管理器"), variant="primary")
					with gr.Row():
						model_type_dropdown = gr.Dropdown(label=i18n("选择模型类型"), choices=MODEL_CHOICES, scale=1)
						download_model_name_dropdown = gr.Dropdown(label=i18n("选择模型"), choices=[i18n("请先选择模型类型")], scale=3)
					model_info = gr.TextArea(label=i18n("模型信息"), value=i18n("请先选择模型"), interactive=False)
					open_model_dir = gr.Button(i18n("打开模型目录"))
					with gr.Row():
						download_button = gr.Button(i18n("自动下载"), variant="primary")
						manual_download_button = gr.Button(i18n("手动下载"), variant="primary")
					output_message_download = gr.Textbox(label="Output Message")
				with gr.Column(scale=1):
					gr.Markdown(i18n("### 注意事项"))
					gr.Markdown(
						value=i18n(
							"1. MSST模型默认下载在pretrain/<模型类型>文件夹下。UVR模型默认下载在设置中的UVR模型目录中。<br>2. 下加载进度可以打开终端查看。如果一直卡着不动或者速度很慢, 在确信网络正常的情况下请尝试重启WebUI。<br>3. 若下载失败, 会在模型目录**留下一个损坏的模型**, 请**务必**打开模型目录手动删除! <br>4. 点击“重启WebUI”按钮后, 会短暂性的失去连接, 随后会自动开启一个新网页。"
						)
					)
					gr.Markdown(i18n("### 模型下载链接"))
					gr.Markdown(i18n("1. 自动从Github, Huggingface或镜像站下载模型。<br>2. 你也可以在此整合包下载链接中的All_Models文件夹中找到所有可用的模型并下载。"))
					gr.Markdown(value=i18n("若自动下载出现报错或下载过慢, 请点击手动下载, 跳转至下载链接。手动下载完成后, 请根据你选择的模型类型放置到对应文件夹内。"))
					gr.Markdown(value=i18n("### 当前UVR模型目录: ") + f"`{uvr_model_folder}`" + i18n(", 如需更改, 请前往设置页面。"))
					gr.Markdown(value=i18n("### 模型安装完成后, 需重启WebUI刷新模型列表"))
					restart_webui = gr.Button(i18n("重启WebUI"), variant="primary")
		with gr.TabItem(label=i18n("安装非官方MSST模型")):
			gr.Markdown(
				value=i18n(
					"你可以从其他途径获取非官方MSST模型, 在此页面完成配置文件设置后, 即可正常使用。<br>注意: 仅支持'.ckpt', '.th', '.chpt'格式的模型。模型显示名字为模型文件名。<br>选择模型类型: 共有三个可选项。依次代表人声相关模型, 多音轨分离模型, 单音轨分离模型。仅用于区分模型大致类型, 可任意选择。<br>选择模型类别: 此选项关系到模型是否能正常推理使用, 必须准确选择!"
				)
			)
			with gr.Row():
				unmsst_model = gr.File(label=i18n("上传非官方MSST模型"), type="filepath")
				unmsst_config = gr.File(label=i18n("上传非官方MSST模型配置文件"), type="filepath")
			with gr.Row():
				unmodel_class = gr.Dropdown(label=i18n("选择模型类型"), choices=["vocal_models", "multi_stem_models", "single_stem_models"], interactive=True)
				unmodel_type = gr.Dropdown(label=i18n("选择模型类别"), choices=MODEL_TYPE, interactive=True)
				unmsst_model_link = gr.Textbox(label=i18n("模型下载链接 (非必须，若无，可跳过)"), value="", interactive=True, scale=2)
			unmsst_model_install = gr.Button(i18n("安装非官方MSST模型"), variant="primary")
			output_message_unmsst = gr.Textbox(label="Output Message")
		with gr.TabItem(label=i18n("安装非官方VR模型")):
			gr.Markdown(value=i18n("你可以从其他途径获取非官方UVR模型, 在此页面完成配置文件设置后, 即可正常使用。<br>注意: 仅支持'.pth'格式的模型。模型显示名字为模型文件名。"))
			with gr.Row():
				unvr_model = gr.File(label=i18n("上传非官方VR模型"), type="filepath")
				with gr.Column():
					with gr.Row():
						unvr_primary_stem = gr.Textbox(label=i18n("主要音轨名称"), value="", interactive=True)
						unvr_secondary_stem = gr.Textbox(label=i18n("次要音轨名称"), value="", interactive=True)
					model_param = gr.Dropdown(label=i18n("选择模型参数"), choices=get_all_model_param(), interactive=True)
					with gr.Row():
						is_karaoke_model = gr.Checkbox(label=i18n("是否为Karaoke模型"), value=False, interactive=True)
						is_BV_model = gr.Checkbox(label=i18n("是否为BV模型"), value=False, interactive=True)
						is_VR51_model = gr.Checkbox(label=i18n("是否为VR 5.1模型"), value=False, interactive=True)
			balance_value = gr.Number(label="balance_value", value=0.0, minimum=0.0, maximum=0.9, step=0.1, interactive=True, visible=False)
			with gr.Row():
				out_channels = gr.Number(label="Out Channels", value=32, minimum=1, step=1, interactive=True, visible=False)
				out_channels_lstm = gr.Number(label="Out Channels (LSTM layer)", value=128, minimum=1, step=1, interactive=True, visible=False)
			upload_param = gr.File(label=i18n("上传参数文件"), type="filepath", interactive=True, visible=False)
			unvr_model_link = gr.Textbox(label=i18n("模型下载链接 (非必须，若无，可跳过)"), value="", interactive=True)
			unvr_model_install = gr.Button(i18n("安装非官方VR模型"), variant="primary")
			output_message_unvr = gr.Textbox(label="Output Message")

	model_type_dropdown.change(fn=upgrade_download_model_name, inputs=model_type_dropdown, outputs=download_model_name_dropdown)
	download_button.click(fn=download_model, inputs=[model_type_dropdown, download_model_name_dropdown], outputs=output_message_download)
	manual_download_button.click(fn=manual_download_model, inputs=[model_type_dropdown, download_model_name_dropdown], outputs=output_message_download)
	is_BV_model.change(fn=update_vr_param, inputs=[is_BV_model, is_VR51_model, model_param], outputs=[balance_value, out_channels, out_channels_lstm, upload_param])
	is_VR51_model.change(fn=update_vr_param, inputs=[is_BV_model, is_VR51_model, model_param], outputs=[balance_value, out_channels, out_channels_lstm, upload_param])
	model_param.change(fn=update_vr_param, inputs=[is_BV_model, is_VR51_model, model_param], outputs=[balance_value, out_channels, out_channels_lstm, upload_param])
	unmsst_model_install.click(fn=install_unmsst_model, inputs=[unmsst_model, unmsst_config, unmodel_class, unmodel_type, unmsst_model_link], outputs=output_message_unmsst)
	unvr_model_install.click(
		fn=install_unvr_model,
		inputs=[
			unvr_model,
			unvr_primary_stem,
			unvr_secondary_stem,
			model_param,
			is_karaoke_model,
			is_BV_model,
			is_VR51_model,
			balance_value,
			out_channels,
			out_channels_lstm,
			upload_param,
			unvr_model_link,
		],
		outputs=output_message_unvr,
	)
	download_model_name_dropdown.change(fn=show_model_info, inputs=[model_type_dropdown, download_model_name_dropdown], outputs=model_info)
	open_model_dir.click(open_model_folder, inputs=model_type_dropdown)
	open_downloadmanager.click(open_download_manager)
	restart_webui.click(webui_restart)
