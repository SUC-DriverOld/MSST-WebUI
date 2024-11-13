import os
import sys
from qfluentwidgets import setTheme, setThemeColor, Theme
from download_manager import DownloadManager
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTranslator
from common.config import cfg


def main():
    app = QApplication(sys.argv)
    translator = QTranslator()
    
    theme = cfg.get(cfg.theme)
    theme_color = cfg.get(cfg.themeColor)
    locale = cfg.get(cfg.language).value
    print(locale)
    translator.load(locale, "app", "./resource/i18n", )
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
    
    
if __name__ == '__main__':
    main()    
