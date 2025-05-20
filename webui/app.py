__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

"""
This file defines the main web UI application for the Music-Source-Separation-Training-Inference-Webui. It utilizes Gradio
to create an interface for separating, training, and configuring models related to music source separation.

Note:
- This source file is publicly accessible only within the source code and is not visible in the packed executable version.
In the packed version, this script is compiled to a `app.cp310-win_amd64.pyd` file, making the source code inaccessible.
- **DO NOT** try to change the code in this file! It is compiled and should not be modified.

This file and its functions collectively define a versatile web interface that consolidates a wide range of functionalities
for music source separation and model management.
"""


import gradio as gr

from utils.constant import WEBUI_CONFIG, LANGUAGE, PACKAGE_VERSION
from webui.utils import i18n, load_configs
from webui import ui

webui_config = load_configs(WEBUI_CONFIG)
language_dict = load_configs(LANGUAGE)


def app(platform, device, force_cpu, theme="tools/themes/theme_blue.json"):
	with gr.Blocks(theme=gr.Theme.load(theme), title="MSST WebUI") as webui:
		gr.Markdown(value=f"""### Music-Source-Separation-Training-Inference-Webui v{PACKAGE_VERSION}""")
		gr.Markdown(
			value=i18n(
				"仅供个人娱乐和非商业用途, 禁止用于血腥/暴力/性相关/政治相关内容。[点击前往教程文档](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)<br>本整合包完全免费, 严禁以任何形式倒卖, 如果你从任何地方**付费**购买了本整合包, 请**立即退款**。<br> 整合包作者: [bilibili@阿狸不吃隼舞](https://space.bilibili.com/403335715) [Github@KitsuneX07](https://github.com/KitsuneX07) | [Bilibili@Sucial](https://space.bilibili.com/445022409) [Github@SUC-DriverOld](https://github.com/SUC-DriverOld) | Gradio主题: [Gradio Theme](https://huggingface.co/spaces/NoCrypt/miku)"
			)
		)

		with gr.Tabs():
			with gr.TabItem(label=i18n("MSST分离")):
				ui.msst(webui_config, device, force_cpu)
			with gr.TabItem(label=i18n("UVR分离")):
				ui.vr(webui_config, force_cpu)
			with gr.TabItem(label=i18n("预设流程")):
				ui.preset(webui_config, force_cpu)
			with gr.TabItem(label=i18n("合奏模式")):
				ui.ensemble(webui_config, force_cpu)
			with gr.TabItem(label=i18n("小工具")):
				ui.tools(webui_config)
			with gr.TabItem(label=i18n("安装模型")):
				ui.models(webui_config)
			with gr.TabItem(label=i18n("MSST训练")):
				ui.train(webui_config, device)
			with gr.TabItem(label=i18n("设置")):
				ui.settings(webui_config, language_dict, platform, device)
	return webui
