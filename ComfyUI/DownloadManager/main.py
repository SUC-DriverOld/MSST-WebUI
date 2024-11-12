import os
import sys
from qfluentwidgets import setTheme, setThemeColor, Theme
from download_manager import DownloadManager
from PySide6.QtWidgets import QApplication
from common.config import cfg


def main():
    # os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    app = QApplication(sys.argv)
    theme = cfg.get(cfg.theme)
    theme_color = cfg.get(cfg.themeColor)
    
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
