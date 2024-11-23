import subprocess

command = "python -m nuitka --standalone --onefile --follow-imports --output-dir=./build "
command += "--windows-disable-console "
command += "--windows-icon-from-ico=./ComfyUI/DownloadManager/resource/icon/DownloadManager.ico "
command += "--output-filename=DownloadManager "
command += "--include-data-files=./ComfyUI/DownloadManager/resource/icon/*=resource/icon/ "
command += "--include-data-files=./ComfyUI/DownloadManager/resource/i18n/*=resource/i18n/ "
command += "--enable-plugin=pyside6 "
command += "./ComfyUI/DownloadManager/main.py"

print(command)

subprocess.run(command, shell=True)
