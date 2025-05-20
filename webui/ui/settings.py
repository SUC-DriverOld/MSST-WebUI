__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr

from utils.constant import *
from webui.utils import i18n, webui_restart, select_folder, log_level_debug
from webui.settings import (
	reset_settings,
	reset_webui_config,
	save_uvr_modeldir,
	check_webui_update,
	webui_goto_github,
	change_language,
	change_download_link,
	change_share_link,
	change_local_link,
	save_port_to_config,
	save_auto_clean_cache,
	save_audio_setting_fn,
	update_rename_model_name,
	change_theme,
)


def settings(webui_config, language_dict, platform, device):
	device = [value for _, value in device.items()]

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
		with gr.TabItem(label=i18n("WebUI设置")):
			with gr.Row():
				with gr.Column(scale=3):
					with gr.Row():
						gr.Textbox(label=i18n("GPU信息"), value=device if len(device) > 1 else device[0], interactive=False)
						gr.Textbox(label=i18n("系统信息"), value=platform, interactive=False)
					with gr.Row():
						set_webui_port = gr.Number(label=i18n("设置WebUI端口, 0为自动"), value=webui_config["settings"].get("port", 0), interactive=True)
						set_language = gr.Dropdown(label=i18n("选择语言"), choices=language_dict.keys(), value=language, interactive=True)
						set_download_link = gr.Dropdown(
							label=i18n("选择MSST模型下载链接"),
							choices=["Auto", i18n("huggingface.co (需要魔法)"), i18n("hf-mirror.com (镜像站可直连)")],
							value=webui_config["settings"]["download_link"] if webui_config["settings"]["download_link"] else "Auto",
							interactive=True,
						)
						set_theme = gr.Dropdown(
							label=i18n("选择WebUI主题"), choices=theme_choices, value=webui_config["settings"]["theme"] if webui_config["settings"]["theme"] else "theme_blue.json", interactive=True
						)
					with gr.Row():
						open_local_link = gr.Checkbox(
							label=i18n("对本地局域网开放WebUI: 开启后, 同一局域网内的设备可通过'本机IP:端口'的方式访问WebUI。"), value=webui_config["settings"]["local_link"], interactive=True
						)
						open_share_link = gr.Checkbox(
							label=i18n("开启公共链接: 开启后, 他人可通过公共链接访问WebUI。链接有效时长为72小时。"), value=webui_config["settings"]["share_link"], interactive=True
						)
						auto_clean_cache = gr.Checkbox(label=i18n("自动清理缓存: 开启后, 每次启动WebUI时会自动清理缓存。"), value=webui_config["settings"]["auto_clean_cache"], interactive=True)
						debug_mode = gr.Checkbox(label=i18n("全局调试模式: 向开发者反馈问题时请开启。(该选项支持热切换)"), value=webui_config["settings"]["debug"], interactive=True)
					with gr.Row():
						select_uvr_model_dir = gr.Textbox(
							label=i18n("选择UVR模型目录"),
							value=webui_config["settings"]["uvr_model_dir"] if webui_config["settings"]["uvr_model_dir"] else "pretrain/VR_Models",
							interactive=True,
							scale=4,
						)
						select_uvr_model_dir_button = gr.Button(i18n("选择文件夹"), scale=1)
					with gr.Row():
						update_message = gr.Textbox(label=i18n("检查更新"), value=i18n("当前版本: ") + PACKAGE_VERSION + i18n(", 请点击检查更新按钮"), interactive=False, scale=3)
						check_update = gr.Button(i18n("检查更新"), scale=1)
						goto_github = gr.Button(i18n("前往Github瞅一眼"))
					with gr.Row():
						reset_all_webui_config = gr.Button(i18n("重置WebUI路径记录"), variant="primary")
						reset_seetings = gr.Button(i18n("重置WebUI设置"), variant="primary")
					setting_output_message = gr.Textbox(label="Output Message")
				with gr.Column(scale=1):
					gr.Markdown(i18n("### 选择UVR模型目录"))
					gr.Markdown(
						i18n(
							"如果你的电脑中有安装UVR5, 你不必重新下载一遍UVR5模型, 只需在下方“选择UVR模型目录”中选择你的UVR5模型目录, 定位到models/VR_Models文件夹。<br>例如: E:/Program Files/Ultimate Vocal Remover/models/VR_Models 点击保存设置或重置设置后, 需要重启WebUI以更新。"
						)
					)
					gr.Markdown(i18n("### 检查更新"))
					gr.Markdown(i18n("从Github检查更新, 需要一定的网络要求。点击检查更新按钮后, 会自动检查是否有最新版本。你可以前往此整合包的下载链接或访问Github仓库下载最新版本。"))
					gr.Markdown(i18n("### 重置WebUI路径记录"))
					gr.Markdown(i18n("将所有输入输出目录重置为默认路径, 预设/模型/配置文件以及上面的设置等**不会重置**, 无需担心。重置WebUI设置后, 需要重启WebUI。"))
					gr.Markdown(i18n("### 重置WebUI设置"))
					gr.Markdown(i18n("仅重置WebUI设置, 例如UVR模型路径, WebUI端口等。重置WebUI设置后, 需要重启WebUI。"))
					gr.Markdown(i18n("### 重启WebUI"))
					gr.Markdown(i18n("点击 “重启WebUI” 按钮后, 会短暂性的失去连接, 随后会自动开启一个新网页。"))
					restart_webui = gr.Button(i18n("重启WebUI"), variant="primary")
		with gr.TabItem(label=i18n("音频输出设置")):
			gr.Markdown(i18n("此页面支持用户自定义修改MSST/VR推理后输出音频的质量。输出音频的**采样率, 声道数与模型支持的参数有关, 无法更改**。<br>修改完成后点击保存设置即可生效。"))
			wav_bit_depth = gr.Radio(
				label=i18n("输出wav位深度"),
				choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"],
				value=webui_config["settings"]["wav_bit_depth"] if webui_config["settings"]["wav_bit_depth"] else "FLOAT",
				interactive=True,
			)
			flac_bit_depth = gr.Radio(
				label=i18n("输出flac位深度"),
				choices=["PCM_16", "PCM_24"],
				value=webui_config["settings"]["flac_bit_depth"] if webui_config["settings"]["flac_bit_depth"] else "PCM_24",
				interactive=True,
			)
			mp3_bit_rate = gr.Radio(
				label=i18n("输出mp3比特率(bps)"),
				choices=["96k", "128k", "192k", "256k", "320k"],
				value=webui_config["settings"]["mp3_bit_rate"] if webui_config["settings"]["mp3_bit_rate"] else "320k",
				interactive=True,
			)
			save_audio_setting = gr.Button(i18n("保存设置"), variant="primary")
			audio_setting_output_message = gr.Textbox(label="Output Message")

	restart_webui.click(fn=webui_restart, outputs=setting_output_message)
	check_update.click(fn=check_webui_update, outputs=update_message)
	goto_github.click(fn=webui_goto_github)
	select_uvr_model_dir_button.click(fn=select_folder, outputs=select_uvr_model_dir)
	reset_seetings.click(fn=reset_settings, outputs=setting_output_message)
	reset_all_webui_config.click(fn=reset_webui_config, outputs=setting_output_message)
	set_webui_port.change(fn=save_port_to_config, inputs=set_webui_port, outputs=setting_output_message)
	auto_clean_cache.change(fn=save_auto_clean_cache, inputs=auto_clean_cache, outputs=setting_output_message)
	select_uvr_model_dir.change(fn=save_uvr_modeldir, inputs=select_uvr_model_dir, outputs=setting_output_message)
	set_language.change(fn=change_language, inputs=set_language, outputs=setting_output_message)
	set_download_link.change(fn=change_download_link, inputs=set_download_link, outputs=setting_output_message)
	open_share_link.change(fn=change_share_link, inputs=open_share_link, outputs=setting_output_message)
	open_local_link.change(fn=change_local_link, inputs=open_local_link, outputs=setting_output_message)
	debug_mode.change(fn=log_level_debug, inputs=debug_mode, outputs=setting_output_message)
	set_theme.change(fn=change_theme, inputs=set_theme, outputs=setting_output_message)
	save_audio_setting.click(fn=save_audio_setting_fn, inputs=[wav_bit_depth, flac_bit_depth, mp3_bit_rate], outputs=audio_setting_output_message)
