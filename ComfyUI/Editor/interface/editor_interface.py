from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget
from PySide6.QtCore import QRect, QPropertyAnimation

from qfluentwidgets import CommandBar, TitleLabel, ScrollArea, PushButton, Action, FluentIcon, Flyout, FlyoutAnimationType
from ComfyUI.Editor.component.editor_view import editorView
from ComfyUI.Editor.component.model_selector import modelSelectorView


class editorInterface(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('editorInterface')
        self.setupUI()

    def setupUI(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setupCommandBar()
        self.setMainView()

    def setupCommandBar(self):
        self.label = TitleLabel(self.tr("Editor"))
        self.command_bar = CommandBar(self)
        self.command_bar.addWidget(self.label)
        self.command_bar.addSeparator()

        self.command_bar.addAction(Action(FluentIcon.ADD, self.tr("Add"), triggered=self.showModelSelector))
        self.command_bar.setFixedHeight(50)
        self.layout.addWidget(self.command_bar)

    def setMainView(self): 
        self.editor_view = editorView(self)
        self.layout.addWidget(self.editor_view)
        self.layout.setStretch(0, 0)
        self.layout.setStretch(1, 1)

    def showModelSelector(self):
        self.model_selector_view = modelSelectorView(self)
        Flyout.make(
            view = self.model_selector_view,
            target = self.command_bar,
            parent = self,
            aniType = FlyoutAnimationType.DROP_DOWN
        )
        