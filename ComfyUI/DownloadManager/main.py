import os
import sys
import shutil

sys.path.append(os.getcwd())

if not os.path.exists("./data"):
	shutil.copytree("data_backup", "data")

from qfluentwidgets import setTheme, setThemeColor, Theme, FluentTranslator
from ComfyUI.DownloadManager.download_manager import DownloadManager
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTranslator
from PySide6.QtGui import QIcon
from ComfyUI.DownloadManager.common.config import cfg


def main():
	app = QApplication(sys.argv)
	app.setWindowIcon(QIcon("ComfyUI/DownloadManager/resource/icon/DownloadManager.ico"))
	translator = QTranslator()

	theme = cfg.get(cfg.theme)
	theme_color = cfg.get(cfg.themeColor)
	locale = cfg.get(cfg.language).value
	fluent_translator = FluentTranslator(locale=locale)

	i18n_path = os.path.join(os.path.dirname(__file__), "resource", "i18n", f"app_{locale.name()}.qm")
	# print(i18n_path)
	translator.load(i18n_path)
	app.installTranslator(translator)
	app.installTranslator(fluent_translator)

	if theme == "Dark":
		setTheme(Theme.DARK)
	elif theme == "Light":
		setTheme(Theme.LIGHT)
	else:
		setTheme(Theme.AUTO)

	setThemeColor(theme_color)
	window = DownloadManager()
	window.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
