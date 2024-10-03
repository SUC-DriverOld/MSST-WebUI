import gradio as gr

from tools.webUI.constant import *
from tools.webUI.utils import i18n, get_device, get_platform, webui_restart, select_folder
from tools.webUI.settings import *

def settings(webui_config, language_dict):
    with gr.Tabs():
        with gr.TabItem(label=i18n("WebUI设置")):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        gpu_list = gr.Textbox(label=i18n("GPU信息"), value=get_device() if len(get_device()) > 1 else get_device()[0], interactive=False)
                        plantform_info = gr.Textbox(label=i18n("系统信息"), value=get_platform(), interactive=False)
                    with gr.Row():
                        set_webui_port = gr.Number(label=i18n("设置WebUI端口, 0为自动"), value=webui_config["settings"].get("port", 0), interactive=True)
                        set_language = gr.Dropdown(label=i18n("选择语言"), choices=language_dict.keys(), value=get_language(), interactive=True)
                        set_download_link = gr.Dropdown(label=i18n("选择MSST模型下载链接"), choices=["Auto", i18n("huggingface.co (需要魔法)"), i18n("hf-mirror.com (镜像站可直连)")], value=webui_config['settings']['download_link'] if webui_config['settings']['download_link'] else "Auto", interactive=True)
                    with gr.Row():
                        open_local_link = gr.Checkbox(label=i18n("对本地局域网开放WebUI: 开启后, 同一局域网内的设备可通过'本机IP:端口'的方式访问WebUI。"), value=webui_config['settings']['local_link'], interactive=True)
                        open_share_link = gr.Checkbox(label=i18n("开启公共链接: 开启后, 他人可通过公共链接访问WebUI。链接有效时长为72小时。"), value=webui_config['settings']['share_link'], interactive=True)
                        auto_clean_cache = gr.Checkbox(label=i18n("自动清理缓存: 开启后, 每次启动WebUI时会自动清理缓存。"), value=webui_config['settings']['auto_clean_cache'], interactive=True)
                    with gr.Row():
                        select_uvr_model_dir = gr.Textbox(label=i18n("选择UVR模型目录"),value=webui_config['settings']['uvr_model_dir'] if webui_config['settings']['uvr_model_dir'] else "pretrain/VR_Models",interactive=True,scale=4)
                        select_uvr_model_dir_button = gr.Button(i18n("选择文件夹"), scale=1)
                    with gr.Row():
                        update_message = gr.Textbox(label=i18n("检查更新"), value=i18n("当前版本: ") + PACKAGE_VERSION + i18n(", 请点击检查更新按钮"), interactive=False,scale=3)
                        check_update = gr.Button(i18n("检查更新"), scale=1)
                        goto_github = gr.Button(i18n("前往Github瞅一眼"))
                    with gr.Row():
                        reset_all_webui_config = gr.Button(i18n("重置WebUI路径记录"), variant="primary")
                        reset_seetings = gr.Button(i18n("重置WebUI设置"), variant="primary")
                    setting_output_message = gr.Textbox(label="Output Message")
                with gr.Column(scale=1):
                    gr.Markdown(i18n("### 设置说明"))
                    gr.Markdown(i18n("### 选择UVR模型目录"))
                    gr.Markdown(i18n("如果你的电脑中有安装UVR5, 你不必重新下载一遍UVR5模型, 只需在下方“选择UVR模型目录”中选择你的UVR5模型目录, 定位到models/VR_Models文件夹。<br>例如: E:/Program Files/Ultimate Vocal Remover/models/VR_Models 点击保存设置或重置设置后, 需要重启WebUI以更新。"))
                    gr.Markdown(i18n("### 检查更新"))
                    gr.Markdown(i18n("从Github检查更新, 需要一定的网络要求。点击检查更新按钮后, 会自动检查是否有最新版本。你可以前往此整合包的下载链接或访问Github仓库下载最新版本。"))
                    gr.Markdown(i18n("### 重置WebUI路径记录"))
                    gr.Markdown(i18n("将所有输入输出目录重置为默认路径, 预设/模型/配置文件以及上面的设置等**不会重置**, 无需担心。重置WebUI设置后, 需要重启WebUI。"))
                    gr.Markdown(i18n("### 重置WebUI设置"))
                    gr.Markdown(i18n("仅重置WebUI设置, 例如UVR模型路径, WebUI端口等。重置WebUI设置后, 需要重启WebUI。"))
                    gr.Markdown(i18n("### 重启WebUI"))
                    gr.Markdown(i18n("点击 “重启WebUI” 按钮后, 会短暂性的失去连接, 随后会自动开启一个新网页。"))
        with gr.TabItem(label=i18n("模型改名")):
            gr.Markdown(i18n("此页面支持用户自定义修改模型名字, 以便记忆和使用。修改完成后, 需要重启WebUI以刷新模型列表。<br>【注意】此操作不可逆 (无法恢复至默认命名), 请谨慎命名。输入新模型名字时, 需保留后缀!"))
            with gr.Row():
                rename_model_type = gr.Dropdown(label=i18n("选择模型类型"), choices=MODEL_CHOICES, value=None, interactive=True, scale=1)
                rename_model_name = gr.Dropdown(label=i18n("选择模型"), choices=[i18n("请先选择模型类型")], value=i18n("请先选择模型类型"), interactive=True, scale=4)
            rename_new_name = gr.Textbox(label=i18n("新模型名"), placeholder=i18n("请输入新模型名字, 需保留后缀!"), interactive=True)
            rename_model = gr.Button(i18n("确认修改"), variant="primary")
            rename_output_message = gr.Textbox(label="Output Message")
    restart_webui = gr.Button(i18n("重启WebUI"), variant="primary")

    restart_webui.click(fn=webui_restart, outputs=setting_output_message)
    check_update.click(fn=check_webui_update, outputs=update_message)
    goto_github.click(fn=webui_goto_github)
    select_uvr_model_dir_button.click(fn=select_folder, outputs=select_uvr_model_dir)
    reset_seetings.click(fn=reset_settings,inputs=[],outputs=setting_output_message)
    reset_all_webui_config.click(fn=reset_webui_config,outputs=setting_output_message)
    set_webui_port.change(fn=save_port_to_config,inputs=[set_webui_port],outputs=setting_output_message)
    auto_clean_cache.change(fn=save_auto_clean_cache,inputs=[auto_clean_cache],outputs=setting_output_message)
    select_uvr_model_dir.change(fn=save_uvr_modeldir,inputs=[select_uvr_model_dir],outputs=setting_output_message)
    set_language.change(fn=change_language,inputs=[set_language],outputs=setting_output_message)
    set_download_link.change(fn=change_download_link,inputs=[set_download_link],outputs=setting_output_message)
    open_share_link.change(fn=change_share_link,inputs=[open_share_link],outputs=setting_output_message)
    open_local_link.change(fn=change_local_link,inputs=[open_local_link],outputs=setting_output_message)
    rename_model_type.change(fn=update_rename_model_name, inputs=[rename_model_type], outputs=[rename_model_name])
    rename_model.click(fn=rename_name, inputs=[rename_model_type, rename_model_name, rename_new_name], outputs=[rename_output_message, rename_model_type, rename_model_name])