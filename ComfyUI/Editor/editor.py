from qfluentwidgets import FluentWindow, NavigationItemPosition
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.Editor.interface.settings_interface import SettingsInterface
from ComfyUI.Editor.interface.editor_interface import EditorInterface
from ComfyUI.Editor.interface.manager_interface import ManagerInterface
from ComfyUI.Editor.interface.logger_interface import LogInterface

class Editor(FluentWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addInterfaces()
        self.stackedWidget.currentChanged.connect(self.onInterfaceChanged)
        self.showMaximized()

    def addInterfaces(self):
        self.editor_interface = EditorInterface(self)
        self.addSubInterface(self.editor_interface, FIF.EDIT, "Editor", NavigationItemPosition.TOP)
        self.manager_interface = ManagerInterface(self)
        self.addSubInterface(self.manager_interface, FIF.SPEED_MEDIUM, "Manager", NavigationItemPosition.TOP)
        self.logger_interface = LogInterface(self)
        self.addSubInterface(self.logger_interface, FIF.DOCUMENT, "Log", NavigationItemPosition.TOP)
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