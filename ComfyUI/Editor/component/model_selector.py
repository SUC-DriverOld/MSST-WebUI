import sys
from qfluentwidgets import TreeWidget, FlyoutViewBase
from PySide6.QtWidgets import QTreeWidgetItem, QSizePolicy, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from ComfyUI.Editor.common.data import models_info

class modelLibraryTree(TreeWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setHeaderHidden(True)
        self.setIndentation(20)
        self.setContentsMargins(0, 0, 0, 0)
        self.buildTree()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def buildTree(self):
        self.addTopLevelItems([
            QTreeWidgetItem(["multi_stem_models"]),
            QTreeWidgetItem(["single_stem_models"]),
            QTreeWidgetItem(["vocal_models"]),
            QTreeWidgetItem(["VR_Models"])
        ])

        for model in models_info:
            model_class = models_info[model]["model_class"]
            child = QTreeWidgetItem([model])
            if not models_info[model]["is_installed"]:
                child.setForeground(0, QColor(127, 127,127))
            match model_class:
                case "multi_stem_models":
                    self.topLevelItem(0).addChild(child)
                case "single_stem_models":
                    self.topLevelItem(1).addChild(child)
                case "vocal_models":
                    self.topLevelItem(2).addChild(child)
                case "VR_Models":
                    self.topLevelItem(3).addChild(child)
                case _:
                    pass

class modelSelectorView(FlyoutViewBase):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.tree = modelLibraryTree(self)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.tree)
        self.setFixedSize(400, 600)