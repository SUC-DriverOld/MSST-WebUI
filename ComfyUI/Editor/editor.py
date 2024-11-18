from qfluentwidgets import FluentWindow, NavigationItemPosition
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.Editor.interface.settings_interface import settingsInterface
from ComfyUI.Editor.interface.editor_interface import editorInterface


class Editor(FluentWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addInterfaces()
        self.showFullScreen()

    def addInterfaces(self):
        self.editor_interface = editorInterface(self)
        self.addSubInterface(self.editor_interface, FIF.EDIT, "Editor", NavigationItemPosition.TOP)
        self.settings_interface = settingsInterface(self)
        self.addSubInterface(self.settings_interface, FIF.SETTING, "Settings", NavigationItemPosition.BOTTOM)