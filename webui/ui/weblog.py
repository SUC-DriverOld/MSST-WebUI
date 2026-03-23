import os
import gradio as gr
import time
import threading
import re
import logging
from webui.utils import i18n, open_folder


log_file_name = os.environ.get("MSST_LOG_FILE", None)
log_file_path = os.path.join("logs", log_file_name) if log_file_name else None
min_level = logging.INFO
log_cache = list()
last_file_position = 0
is_start = False

LEVEL_PATTERN = re.compile(r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]")
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def update_log_cache():
    global log_cache, last_file_position
    while True:
        if not log_file_path or not os.path.exists(log_file_path):
            time.sleep(1)
            continue
        try:
            current_size = os.path.getsize(log_file_path)
            if current_size < last_file_position:
                last_file_position = 0
            if current_size > last_file_position:
                with open(log_file_path, "r", encoding="utf-8") as f:
                    f.seek(last_file_position)
                    new_lines = f.read().split("\n")[:-1] # 去掉最后一行空行
                last_file_position = current_size
                if new_lines:
                    log_cache.extend(new_lines)
        except Exception as e:
            print(f"Error updating log cache: {e}")
        time.sleep(1)

threading.Thread(target=update_log_cache, daemon=True).start()

def filter(log_cache):
    def extract_level(line):
        match = LEVEL_PATTERN.search(line)
        if match:
            return LOG_LEVELS[match.group(1)]
        return logging.CRITICAL
    if not log_cache:
        return []
    return [line for line in log_cache if extract_level(line) >= min_level]

def start_logging(log_level):
    global min_level, is_start
    if is_start: return
    is_start = True
    min_level = LOG_LEVELS.get(log_level, logging.INFO)
    while True:
        filtered_lines = filter(log_cache)
        yield "\n".join(filtered_lines)
        time.sleep(1)

def set_log_level(log_level):
    global min_level
    min_level = LOG_LEVELS.get(log_level, logging.INFO)
    filtered_lines = filter(log_cache)
    return "\n".join(filtered_lines)

def weblog():
    with gr.TabItem(label=i18n("日志")) as log_tab:
        log_level = gr.Radio(
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            label=i18n("日志等级"),
            value="INFO",
            interactive=True,
        )
        log_file = gr.TextArea(
            label=i18n("日志信息"),
            value="",
            lines=30,
            interactive=False,
            autoscroll=True
        )
        open_log_folder = gr.Button(value=i18n("打开日志文件夹"), variant="primary")
        log_tab.select(fn=start_logging, inputs=log_level, outputs=log_file)
        log_level.change(fn=set_log_level, inputs=log_level, outputs=log_file)
        open_log_folder.click(fn=open_folder, inputs=gr.Textbox(value="logs", visible=False))