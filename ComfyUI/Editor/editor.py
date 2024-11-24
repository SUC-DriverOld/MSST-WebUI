from qfluentwidgets import FluentWindow, NavigationItemPosition
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.Editor.interface.settings_interface import SettingsInterface
from ComfyUI.Editor.interface.editor_interface import EditorInterface


class Editor(FluentWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addInterfaces()
        self.showMaximized()

    def addInterfaces(self):
        self.editor_interface = EditorInterface(self)
        self.addSubInterface(self.editor_interface, FIF.EDIT, "Editor", NavigationItemPosition.TOP)
        self.settings_interface = SettingsInterface(self)
        self.addSubInterface(self.settings_interface, FIF.SETTING, "Settings", NavigationItemPosition.BOTTOM)