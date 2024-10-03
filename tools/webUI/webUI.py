import gradio as gr

from tools.webUI.constant import *
from tools.webUI.utils import i18n, load_configs
import tools.webUI.ui as ui

webui_config = load_configs(WEBUI_CONFIG)
presets = load_configs(PRESETS)
language_dict = load_configs(LANGUAGE)

def app():
    with gr.Blocks(
            theme=gr.Theme.load('tools/themes/theme_schema@1.2.2.json')
    ) as app:
        gr.Markdown(value=f"""### Music-Source-Separation-Training-Inference-Webui v{PACKAGE_VERSION}""")
        gr.Markdown(value=i18n("仅供个人娱乐和非商业用途, 禁止用于血腥/暴力/性相关/政治相关内容。[点击前往教程文档](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)<br>本整合包完全免费, 严禁以任何形式倒卖, 如果你从任何地方**付费**购买了本整合包, 请**立即退款**。<br> 整合包作者: [bilibili@阿狸不吃隼舞](https://space.bilibili.com/403335715) [Github@KitsuneX07](https://github.com/KitsuneX07) | [Bilibili@Sucial](https://space.bilibili.com/445022409) [Github@SUC-DriverOld](https://github.com/SUC-DriverOld) | Gradio主题: [Gradio Theme](https://huggingface.co/spaces/NoCrypt/miku)"))

        with gr.Tabs():
            with gr.TabItem(label=i18n("MSST分离")):
                ui.msst(webui_config)

            with gr.TabItem(label=i18n("UVR分离")):
                ui.vr(webui_config)

            with gr.TabItem(label=i18n("预设流程")):
                ui.preset(webui_config, presets)

            with gr.TabItem(label=i18n("小工具")):
                ui.tools(webui_config)

            with gr.TabItem(label=i18n("安装模型")):
                ui.models(webui_config)

            with gr.TabItem(label=i18n("MSST训练")):
                ui.train(webui_config)

            with gr.TabItem(label=i18n("设置")):
                ui.settings(webui_config, language_dict)

    return app