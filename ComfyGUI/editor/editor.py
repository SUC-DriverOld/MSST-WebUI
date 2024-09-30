from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem, QApplication
from PySide6.QtGui import QFont, QDrag
from PySide6.QtCore import Qt, QMimeData, QByteArray, QPoint
from qfluentwidgets import setTheme, Theme
from view import ComfyGUIView
from scene import ComfyGUIScene
from nodes.model_node import MSSTModelNode, VRModelNode
from nodes.data_flow_node import InputNode, OutputNode

import json, os


class ComfyGUIEditor(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.node_position_offset = 100
        current_dir = os.path.dirname(__file__)
        msst_model_path = os.path.join(current_dir, '..', '..', 'data', 'msst_model_map.json')
        vr_model_path = os.path.join(current_dir, '..', '..', 'data', 'vr_model_map.json')
        self.msst_model_data = self.load_json(msst_model_path)
        self.vr_model_data = self.load_json(vr_model_path)
        self.setup_editor()

    def load_json(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def setup_editor(self):
        # 设置窗口大小和标题
        self.setGeometry(300, 200, 720, 720)
        self.setWindowTitle("ComfyMSS Editor")

        # 创建主布局
        self.layout = QVBoxLayout(self)

        # 创建Splitter，用于调整树形控件和场景视图的宽度
        splitter = QSplitter(Qt.Horizontal, self)
        self.layout.addWidget(splitter)

        # 创建场景和视图，并添加到Splitter中
        self.scene = ComfyGUIScene()
        self.view = ComfyGUIView(self.scene, self)
        self.view.setAcceptDrops(True)  # 启用接受拖放
        splitter.addWidget(self.view)

        # 创建树形控件
        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabel("可用模型")
        self.tree.setDragEnabled(True)  # 启用拖放
        splitter.addWidget(self.tree)

        self.tree.itemDoubleClicked.connect(self.add_selected_model_node)  # 双击添加
        self.tree.startDrag = self.start_drag  # 重写startDrag方法

        # 设置字体
        font = QFont("Segoe UI", 12)
        self.tree.setFont(font)
        self.tree.setStyleSheet("""
            QTreeWidget::item {
                height: 30px;
            }
        """)

        self.populate_tree()
        self.show()

    def populate_tree(self):
        # 填充树形控件
        for category, models in self.msst_model_data.items():
            category_item = QTreeWidgetItem(self.tree)
            category_item.setText(0, category)
            category_item.setData(0, Qt.UserRole, category)  # 设置model_class
            for model in models:
                model_item = QTreeWidgetItem(category_item)
                model_item.setText(0, model["name"])
                model["model_class"] = category  # 添加model_class信息到模型数据
                model_item.setData(0, Qt.UserRole, model)

        vr_category_item = QTreeWidgetItem(self.tree)
        vr_category_item.setText(0, "vr_models")
        for model_name, model in self.vr_model_data.items():
            model["model_class"] = "vr_models"
            model["name"] = model_name
            model_item = QTreeWidgetItem(vr_category_item)
            model_item.setText(0, model_name)
            model_item.setData(0, Qt.UserRole, model)

        # 添加输入和输出节点
        io_category_item = QTreeWidgetItem(self.tree)
        io_category_item.setText(0, "输入和输出")

        input_node_item = QTreeWidgetItem(io_category_item)
        input_node_item.setText(0, "Input Node")
        input_node_item.setData(0, Qt.UserRole, "InputNode")

        output_node_item = QTreeWidgetItem(io_category_item)
        output_node_item.setText(0, "Output Node")
        output_node_item.setData(0, Qt.UserRole, "OutputNode")

    def add_selected_model_node(self, item, column):
        # 处理双击事件
        if item.childCount() > 0:
            return
        model_info = item.data(0, Qt.UserRole)
        pos = [100, 100]
        self.view.create_node(model_info, pos)

    def start_drag(self, event):
        item = self.tree.currentItem()
        if not item:
            return
        mime_data = QMimeData()
        model_info = item.data(0, Qt.UserRole)
        mime_data.setData('application/json', QByteArray(json.dumps(model_info).encode('utf-8')))

        drag = QDrag(self)
        drag.setMimeData(mime_data)
        result = drag.exec(Qt.MoveAction)
