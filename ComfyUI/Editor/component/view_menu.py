from qfluentwidgets import RoundMenu, Action, FluentIcon
from PySide6.QtGui import QAction
from ComfyUI.Editor.common.data import models_info

class ViewMenu(RoundMenu):
    def __init__(self, parent=None, scene_pos = None, scene = None):
        super().__init__(parent)
        self.scene = scene
        self.scene_pos = scene_pos
        self.addNodeMenu()

    def addNodeMenu(self):
        self.node_submenu = RoundMenu(self.tr("Add Nodes"))
        self.node_submenu.setIcon(FluentIcon.ADD)
        self.addMenu(self.node_submenu)
        self.addModelMenu()
        self.addInputMenu()
        self.addOutputMenu()

    def addInputMenu(self):
        input_submenu = RoundMenu("Input Nodes")
        input_submenu.addAction(QAction("Input Node", triggered=lambda: self.addNode("Input Node")))
        input_submenu.addAction(QAction("File Input Node", triggered=lambda: self.addNode("File Input Node")))
        self.node_submenu.addMenu(input_submenu)

    def addOutputMenu(self):
        output_submenu = RoundMenu("Output Nodes")
        output_submenu.addAction(QAction("Output Node", triggered=lambda: self.addNode("Output Node")))
        self.node_submenu.addMenu(output_submenu)

    def addModelMenu(self):
        VR_Models_submenu = RoundMenu("VR Models")
        vocal_models_submenu = RoundMenu("Vocal Models")
        single_stem_models_submenu = RoundMenu("Single Stem Models")
        multi_stem_models_submenu = RoundMenu("Multi Stem Models")
        
        self.node_submenu.addMenu(vocal_models_submenu)
        self.node_submenu.addMenu(single_stem_models_submenu)
        self.node_submenu.addMenu(multi_stem_models_submenu)
        self.node_submenu.addMenu(VR_Models_submenu)

        for model in models_info:
            model_class = models_info[model]["model_class"]
            action = QAction(model, triggered=lambda checked, model=model: self.addNode(model))
            match model_class:
                case "VR_Models":
                    VR_Models_submenu.addAction(action)
                case "vocal_models":
                    vocal_models_submenu.addAction(action)
                case "single_stem_models":
                    single_stem_models_submenu.addAction(action)
                case "multi_stem_models":
                    multi_stem_models_submenu.addAction(action)
                case _:
                    pass

    def addNode(self, node_name):
        self.scene.addNodeFromText(node_name, self.scene_pos)
         

            