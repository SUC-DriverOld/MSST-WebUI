from PySide6.QtWidgets import QFrame, QVBoxLayout, QFileDialog
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
import os

from qfluentwidgets import CommandBar, TitleLabel, Action, FluentIcon, InfoBar, InfoBarPosition
from ComfyUI.Editor.component.editor_view import EditorView
# from ComfyUI.Editor.component.model_selector import modelSelectorView


class EditorInterface(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('EditorInterface')
        self.scene = None
        self.setupUI()

    def setupUI(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setMainView()
        self.setupCommandBar()

    def setupCommandBar(self):
        self.label = TitleLabel(self.tr("Editor"))
        self.command_bar = CommandBar(self)
        self.command_bar.addWidget(self.label)
        self.command_bar.addSeparator()
        self.command_bar.setButtonTight(True)
        self.command_bar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        # self.command_bar.addAction(Action(FluentIcon.ADD, self.tr("Add"), triggered=self.showModelSelector))
        self.command_bar.addAction(Action(FluentIcon.FOLDER, self.tr("Open Preset Folder"), triggered=self.openPresetFolder))
        self.command_bar.addAction(Action(FluentIcon.SAVE, self.tr("Save"), triggered=self.savePreset))
        self.command_bar.addAction(Action(FluentIcon.PASTE, self.tr("Load"), triggered=self.loadPreset))
        self.layout.insertWidget(0, self.command_bar)

    def setMainView(self): 
        self.editor_view = EditorView(self)
        self.scene = self.editor_view.scene
        self.layout.addWidget(self.editor_view)
        self.layout.setStretch(0, 0)
        self.layout.setStretch(1, 1)

    # def showModelSelector(self):
    #     self.model_selector_view = modelSelectorView(self)
    #     Flyout.make(
    #         view = self.model_selector_view,
    #         target = self.command_bar,
    #         parent = self,
    #         aniType = FlyoutAnimationType.DROP_DOWN
    #     )

    def openPresetFolder(self):
        default_path = "./ComfyUI/Editor/data/presets"
        absolute_path = os.path.abspath(default_path)
        folder_url = QUrl.fromLocalFile(absolute_path)
        QDesktopServices.openUrl(folder_url)

    def savePreset(self):
        default_path = "./ComfyUI/Editor/data/presets"
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            self.tr("Save Preset"), 
            default_path, 
            "Preset Files (*.preset)", 
            options=QFileDialog.DontUseNativeDialog
        )

        if file_path:
            # 检查文件是否有 .preset 后缀，如果没有则追加
            if not file_path.endswith(".preset"):
                file_path += ".preset"

            self.scene.saveToJson(file_path)
            InfoBar.success(
                title=self.tr("Save Preset Success"),
                content=self.tr(f"Preset saved to {file_path}"),
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )


    def loadPreset(self):
        default_path = "./ComfyUI/Editor/data/presets"
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr("load preset"), default_path, "Preset Files (*.preset)", options=QFileDialog.DontUseNativeDialog)
        if file_path:
            self.scene.loadFromJson(file_path)
            InfoBar.success(
                title=self.tr("Load Preset Success"),
                content=self.tr(f"Preset loaded from {file_path}"),
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            

        