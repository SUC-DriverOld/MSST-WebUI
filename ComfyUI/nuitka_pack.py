import subprocess
import os

PYTHON_PATH = "C:/Users/KitsuneX07/AppData/Local/anaconda3/envs/msst/python.exe"
GITHUB_WORKSPACE = "https://github.com/SUC-DriverOld/MSST-WebUI"

command = f"{PYTHON_PATH} -m nuitka --standalone --onefile --follow-imports --output-dir=./build "
command += "--windows-console-mode=disable "
command += '--copyright="Copyright (c) MSST-WebUI-Develop-Team KitsuneX07. Licensed under GNU Affero General Public License v3.0" '
command += f'--trademark="Free software under AGPL-3.0: Respect copyleft, share your modifications. Also visit {GITHUB_WORKSPACE} for updates and more information" '
command += "--windows-icon-from-ico=./ComfyUI/DownloadManager/resource/icon/DownloadManager.ico "
command += "--output-filename=DownloadManager "
command += "--include-data-files=./ComfyUI/DownloadManager/resource/icon/*=resource/icon/ "
command += "--include-data-files=./ComfyUI/DownloadManager/resource/i18n/*=resource/i18n/ "
command += "--enable-plugin=pyside6 "
command += "--product-version=1.1.0 "
command += "--msvc=latest "
command += "./ComfyUI/DownloadManager/main.py"

print(command)

subprocess.run(command, shell=True)
