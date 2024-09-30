import sys
import os
from PySide6.QtWidgets import QApplication
from editor import ComfyGUIEditor

if __name__ == "__main__":
    app = QApplication([])
    editor = ComfyGUIEditor()
    sys.exit(app.exec())
