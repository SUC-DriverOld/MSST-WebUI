from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget, QHBoxLayout
from PySide6.QtGui import QAction
from PySide6.QtCore import Signal, Qt
from qfluentwidgets import CommandBar, RoundMenu, FluentIcon, ScrollArea, SwitchButton, SpinBox, DoubleSpinBox, CardWidget, BodyLabel, CaptionLabel, TransparentDropDownPushButton, TransparentPushButton
from ComfyUI.Editor.common.data import models_info
import os
import json

class NodeSelectorMenu(RoundMenu):

    selected_model = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        VR_Models_submenu = RoundMenu("VR Models")
        vocal_models_submenu = RoundMenu("Vocal Models")
        single_stem_models_submenu = RoundMenu("Single Stem Models")
        multi_stem_models_submenu = RoundMenu("Multi Stem Models")
        
        self.addMenu(vocal_models_submenu)
        self.addMenu(single_stem_models_submenu)
        self.addMenu(multi_stem_models_submenu)
        self.addMenu(VR_Models_submenu)

        for model in models_info:
            model_class = models_info[model]["model_class"]
            path = models_info[model]["target_position"]
            if not os.path.isfile(path):
                continue
            action = QAction(model, triggered=lambda checked, model=model: self.selected_model.emit(model))
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

class ParameterNumericCard(CardWidget):
    def __init__(self, param_data, parent=None):
        super().__init__(parent)
        self.param_data = param_data
        self.setup_ui()

    def setup_ui(self):
        self.hBoxLayout = QHBoxLayout(self)
        self.vBoxLayout = QVBoxLayout()

        self.setFixedHeight(80)
        self.hBoxLayout.setContentsMargins(20, 15, 20, 15)
        self.hBoxLayout.setSpacing(15)

        self.titleLabel = BodyLabel(self.param_data['parameter'], self)
        
        default_value = self.param_data['default_value']
        self.contentLabel = CaptionLabel(self.tr(f"default value: {default_value} (range: {self.param_data['min_value']} - {self.param_data['max_value']})"), self)

        if self.param_data['type'] == 'int':
            self.spinBox = SpinBox(self)
        else:
            self.spinBox = DoubleSpinBox(self)
            
        self.spinBox.setRange(self.param_data['min_value'], self.param_data['max_value'])
        self.spinBox.setValue(self.param_data['current_value'])
        self.spinBox.setFixedWidth(150)

        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setSpacing(3)
        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.addWidget(self.contentLabel)
        
        self.hBoxLayout.addLayout(self.vBoxLayout)
        self.hBoxLayout.addStretch(1)
        self.hBoxLayout.addWidget(self.spinBox, 0, Qt.AlignVCenter)

    def get_current_value(self):
        return self.spinBox.value()

    def reset_to_default(self):
        self.spinBox.setValue(self.param_data['default_value'])

class ParameterBoolCard(CardWidget):
    def __init__(self, param_data, parent=None):
        super().__init__(parent)
        self.param_data = param_data
        self.setup_ui()

    def setup_ui(self):
        self.hBoxLayout = QHBoxLayout(self)
        self.vBoxLayout = QVBoxLayout()

        self.setFixedHeight(80)
        self.hBoxLayout.setContentsMargins(20, 15, 20, 15)
        self.hBoxLayout.setSpacing(15)

        self.titleLabel = BodyLabel(self.param_data['parameter'], self)
        
        default_state = self.tr("on") if self.param_data['default_value'] else self.tr("off")
        self.contentLabel = CaptionLabel(self.tr(f"default state: {default_state}"), self)
        
        # 开关按钮
        self.switch = SwitchButton(self)
        self.switch.setChecked(self.param_data['current_value'])

        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setSpacing(3)
        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.addWidget(self.contentLabel)
        
        self.hBoxLayout.addLayout(self.vBoxLayout)
        self.hBoxLayout.addStretch(1)
        self.hBoxLayout.addWidget(self.switch, 0, Qt.AlignVCenter)

    def get_current_value(self):
        return self.switch.isChecked()

    def reset_to_default(self):
        self.switch.setChecked(self.param_data['default_value'])

class NodeManagerInterface(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('NodeManagerInterface')
        self.current_model = None
        self.setup_ui()

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(12)

        self.commandBar = CommandBar()
        self.commandBar.setFixedHeight(50)
        
        # 模型选择按钮
        self.modelButton = TransparentDropDownPushButton(FluentIcon.MENU, self.tr("choose model"), self)
        self.modelButton.setFixedHeight(34)
        self.menu = NodeSelectorMenu(self)
        self.menu.selected_model.connect(self.on_model_selected)
        self.modelButton.clicked.connect(self.show_menu)
        
        # 保存按钮
        self.saveButton = TransparentPushButton(FluentIcon.SAVE, self.tr("save"), self)
        self.saveButton.setFixedHeight(34)
        self.saveButton.clicked.connect(self.save_parameters)
        
        # 重置按钮
        self.resetButton = TransparentPushButton(FluentIcon.RETURN, self.tr("reset"), self)
        self.resetButton.setFixedHeight(34)
        self.resetButton.clicked.connect(self.reset_parameters)

        self.commandBar.addWidget(self.modelButton)
        self.commandBar.addWidget(self.saveButton)
        self.commandBar.addWidget(self.resetButton)

        self.scroll = ScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.parameter_layout = QVBoxLayout(self.scroll_content)
        self.parameter_layout.setSpacing(12)
        self.parameter_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll.setWidget(self.scroll_content)
        
        self.main_layout.addWidget(self.commandBar)
        self.main_layout.addWidget(self.scroll)
        
        # 初始禁用保存和重置按钮
        self.saveButton.setEnabled(False)
        self.resetButton.setEnabled(False)

    def save_parameters(self):
        if not self.current_model:
            return
            
        cfg_path = f"./ComfyUI/Editor/data/nodes/{self.current_model}.json"
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        
        # 更新数值参数
        for i in range(self.parameter_layout.count()):
            widget = self.parameter_layout.itemAt(i).widget()
            if isinstance(widget, ParameterNumericCard):
                param_name = widget.param_data['parameter']
                for param in data['parameter']:
                    if param['parameter'] == param_name:
                        param['current_value'] = widget.get_current_value()
                        break
            elif isinstance(widget, ParameterBoolCard):
                param_name = widget.param_data['parameter']
                for param in data['bool']:
                    if param['parameter'] == param_name:
                        param['current_value'] = widget.get_current_value()
                        break
        
        # 保存到文件
        with open(cfg_path, 'w') as f:
            json.dump(data, f, indent=4)

    def reset_parameters(self):
        # 重置所有参数到默认值
        for i in range(self.parameter_layout.count()):
            widget = self.parameter_layout.itemAt(i).widget()
            if isinstance(widget, (ParameterNumericCard, ParameterBoolCard)):
                widget.reset_to_default()

    def on_model_selected(self, model):
        self.current_model = model
        cfg_path = f"./ComfyUI/Editor/data/nodes/{model}.json"
        with open(cfg_path, 'r') as f:
            data = json.load(f)
            self.update_editor(data)
        
        # 启用保存和重置按钮
        self.saveButton.setEnabled(True)
        self.resetButton.setEnabled(True)

    def show_menu(self):
        pos = self.modelButton.mapToGlobal(self.modelButton.rect().bottomLeft())
        self.menu.exec(pos)

    def update_editor(self, data):
        # 清除现有控件
        while self.parameter_layout.count():
            item = self.parameter_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 添加数值参数卡片
        if 'parameter' in data:
            for param in data['parameter']:
                card = ParameterNumericCard(param)
                self.parameter_layout.addWidget(card)
        
        # 添加布尔参数卡片
        if 'bool' in data:
            for bool_param in data['bool']:
                card = ParameterBoolCard(bool_param)
                self.parameter_layout.addWidget(card)

        self.parameter_layout.addStretch()
        
        # 确保UI更新
        self.scroll_content.updateGeometry()
        self.scroll.updateGeometry()
        self.update()    