__license__= "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import webview
import multiprocessing
import warnings
import socket
from tkinter import messagebox
from utils.constant import THEME_FOLDER, PACKAGE_VERSION

multiprocessing.set_start_method('spawn', force=True)
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["MSST_USE_WEBVIEW"] = "1"
warnings.filterwarnings("ignore")


def find_free_port(ip, start_port=11451, end_port=19198):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((ip, port))
                return port
            except OSError:
                continue
    messagebox.showerror("Error", f"Could not find a free port in the range {start_port}-{end_port}.")
    os._exit(1)

def launcher(server_name, server_port):
    import platform
    from webui.utils import i18n, logger
    from webui.setup import setup_webui
    from torch import cuda, backends
    from webui import app

    devices = {}
    force_cpu = False
    if cuda.is_available():
        for i in range(cuda.device_count()):
            devices[f"cuda{i}"] = f"{i}: {cuda.get_device_name(i)}"
        logger.info(i18n("检测到CUDA, 设备信息: ") + str(devices))
    elif backends.mps.is_available():
        devices = {"mps": i18n("使用MPS")}
        logger.info(i18n("检测到MPS, 使用MPS"))
    else:
        devices = {"cpu": i18n("无可用的加速设备, 使用CPU")}
        logger.warning(i18n("\033[33m未检测到可用的加速设备, 使用CPU\033[0m"))
        logger.warning(i18n("\033[33m如果你使用的是NVIDIA显卡, 请更新显卡驱动至最新版后重试\033[0m"))
        force_cpu = True
    platform_info = f"System: {platform.system()}, Machine: {platform.machine()}"
    logger.info(f"WebUI Version: {PACKAGE_VERSION}, {platform_info}")

    webui_config = setup_webui()
    theme_path = os.path.join(THEME_FOLDER, webui_config["settings"].get("theme", "theme_blue.json"))
    logger.debug(f"Launching WebUI with parameters: ip_address={server_name}, port={server_port}")

    app.app(
        platform=platform_info, device=devices, force_cpu=force_cpu, theme=theme_path
    ).queue().launch(
        inbrowser=False, share=False, server_name=server_name, server_port=server_port,
        show_api=False, favicon_path="docs/logo.png"
    )

def start_gradio(server_name, server_port):
    gradio_process = multiprocessing.Process(target=launcher, args=(server_name, server_port))
    gradio_process.start()
    return gradio_process

def get_html(ip, port):
    with open(os.path.join(current_dir, "webui", "webview", "index.html"), 'r', encoding='utf-8') as f:
        html = f.read()
    html = html.replace("{{ip}}", ip)
    html = html.replace("{{port}}", str(port))
    return html

def main():
    server_name = "127.0.0.1"
    server_port = find_free_port(server_name)
    isdebug = os.environ.get("MSST_WEBVIEW_DEBUG", "0") == "1"

    try:
        gradio_process = start_gradio(server_name, server_port)
        html = get_html(server_name, server_port)
        window = webview.create_window(
            title='MSST WebView GUI',
            html=html,
            width=1300,
            height=750,
            frameless=False,
            easy_drag=False,
            text_select=False,
            confirm_close=False
        )
        webview.start(debug=isdebug, http_server=False)
        gradio_process.terminate()
        gradio_process.join()
    except Exception as e:
        import traceback
        messagebox.showerror("Error", f"Failed to start the webview: {e}\n{traceback.format_exc()}")
    os._exit(0)

if __name__ == "__main__":
    main()