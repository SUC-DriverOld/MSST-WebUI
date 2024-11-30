from qfluentwidgets import FluentWindow, NavigationItemPosition
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.Editor.interface.settings_interface import SettingsInterface
from ComfyUI.Editor.interface.editor_interface import EditorInterface


class Editor(FluentWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addInterfaces()
        self.stackedWidget.currentChanged.connect(self.onInterfaceChanged)
        self.showMaximized()

    def addInterfaces(self):
        self.editor_interface = EditorInterface(self)
        self.addSubInterface(self.editor_interface, FIF.EDIT, "Editor", NavigationItemPosition.TOP)
        self.settings_interface = SettingsInterface(self)
        self.addSubInterface(self.settings_interface, FIF.SETTING, "Settings", NavigationItemPosition.BOTTOM)
        
    def onInterfaceChanged(self):
        # 获取当前激活的界面
        current_widget = self.stackedWidget.currentWidget()
        
        # 禁用所有子界面的更新
        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            widget.setUpdatesEnabled(False)
            
        # 只启用当前界面的更新
        current_widget.setUpdatesEnabled(True)    