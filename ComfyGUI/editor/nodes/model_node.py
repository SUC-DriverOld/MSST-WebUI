import sys
import os
import yaml
import json
from ml_collections import ConfigDict
from omegaconf import OmegaConf

from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QColor, QBrush, QFont
from PySide6.QtWidgets import QGraphicsProxyWidget, QComboBox, QLabel, QWidget, QHBoxLayout, QGraphicsDropShadowEffect, \
    QGraphicsTextItem, QGraphicsItem, QVBoxLayout, QButtonGroup, QRadioButton

from config import EditorConfig, NodeConfig
from node_port import InputPort, OutputPort, ParamPort, BoolPort, NodePort
from node import Node
from nodes.data_flow_node import OutputNode, InputNode
from comfy_infer import ComfyMSST, ComfyVR
current_dir = os.path.dirname(__file__)
sys.path.append('..')

TEMP_PATH = "tmpdir"


class ModelNode(Node):

    def __init__(self, model_class = None, model_name = None, model_type = None, input_ports = None, param_ports = None,
                 output_ports = None, bool_ports = None, scene = None, parent = None, upstream_node = None,
                 downstream_nodes = None):
        super().__init__(parent)
        self._model_class = model_class
        self._model_name = model_name
        self._model_type = model_type
        self._scene = scene
        self.input_ports = input_ports or []
        self.param_ports = param_ports or []
        self.output_ports = output_ports or []
        self.bool_ports = bool_ports or []
        self.upstream_node = upstream_node
        self.downstream_nodes = downstream_nodes or []
        self._node_width = self.node_width_min
        self._node_height = self.node_height_min
        self._shadow = QGraphicsDropShadowEffect()
        self._shadow.setOffset(0, 0)
        self._shadow.setBlurRadius(20)
        self._shadow_color = QColor('#aaeeee00')
        self.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.init_node_color()
        self.init_title()
        self.update_ports()
        self.init_output_format()

    def init_node_color(self):
        self._pen_selected = QPen(QColor('#ddffee00'))
        self._brush_background = QBrush(QColor('#dd151515'))
        self._title_bak_color = '#39c5bb'
        title_color = QColor(self._title_bak_color)
        self._pen_default = QPen(title_color)
        title_color.setAlpha(200)
        self._brush_title_back = QBrush(title_color)

    def init_title(self):
        self._title_font_size = EditorConfig.editor_node_title_font_size
        self._title_font = QFont(EditorConfig.editor_node_title_font, self._title_font_size)
        self._title_color = Qt.white

        self._title_line1, self._title_line2 = QGraphicsTextItem(self), QGraphicsTextItem(self)
        self._title_line1.setPlainText(self._model_class)
        self._friendly_name = self._model_name[:30] + '...' if len(self._model_name) > 30 else self._model_name

        self._title_line2.setPlainText(self._friendly_name)
        # for title_line in [self._title_line1, self._title_line2]:
        #     title_line.setFont(self._title_font)
        #     title_line.setDefaultTextColor(self._title_color)

        self._title_line1.setFont(self._title_font)
        self._title_line1.setDefaultTextColor(self._title_color)
        self._title_line2.setFont(QFont(EditorConfig.editor_node_title_font, self._title_font_size - 3))
        self._title_line2.setDefaultTextColor(self._title_color)

        self._title_line1.setPos(self.title_padding, self.title_padding)
        self._title_line2.setPos(self.title_padding, self.title_padding * 3 + self._title_font_size)
        title_width = self._title_font_size * len(self._model_name) + 2 * self.title_padding
        # print(self._node_width, title_width)
        # self._node_width = max(self._node_width, title_width)
        self.title_height = 6 * self.title_padding + 2 * self._title_font_size

    def update_ports(self):
        self.init_ports()
        for port_list in [self.input_ports, self.param_ports, self.bool_ports, self.output_ports]:
            for i, port in enumerate(port_list):
                self.add_port(port, index = i)

    def add_port(self, port: NodePort, index = 0):
        self._node_width = max(self._node_width, port._port_width + self.port_padding * 2)
        self._node_height = self.title_height + (
                max(len(self.input_ports), len(self.output_ports)) + len(self.param_ports) + len(
            self.bool_ports)) * (self.port_padding + port._port_icon_size) + self.port_padding
        port.add_to_parent_node(self, self._scene)

        y_offset = self.title_height + index * (self.port_padding + port._port_icon_size) + self.port_padding
        if port.port_type == NodePort.PORT_TYPE_INPUT:
            port.setPos(self.port_padding, y_offset)
        elif port.port_type == NodePort.PORT_TYPE_OUTPUT:
            port.setPos(self._node_width - port._port_width - self.port_padding, y_offset)
        elif port.port_type == NodePort.PORT_TYPE_PARAM:
            port.setPos(self.port_padding, y_offset + max(len(self.input_ports), len(self.output_ports)) * (
                    self.port_padding + port._port_icon_size))
        elif port.port_type == NodePort.PORT_TYPE_BOOL:
            port.setPos(self.port_padding,
                        y_offset + (len(self.param_ports) + max(len(self.input_ports), len(self.output_ports))) * (
                                self.port_padding + ParamPort()._port_icon_size))

    def init_output_format(self):
        widget = QWidget()
        layout = QHBoxLayout()

        widget.setStyleSheet("background-color: transparent;")

        self.output_format_group = QButtonGroup(widget)

        formats = ['wav', 'flac', 'mp3']
        for fmt in formats:
            radio_button = QRadioButton(fmt)
            radio_button.setStyleSheet("""
                QRadioButton { 
                    color: white; 
                    margin-right: 15px;
                }
                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                }
            """)
            layout.addWidget(radio_button)
            self.output_format_group.addButton(radio_button)

        layout.setAlignment(Qt.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        widget.setLayout(layout)

        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(widget)

        proxy.setPos(self.title_padding * 2, self._node_height)
        self._node_height += widget.sizeHint().height() + self.port_padding
        self._node_width = max(self._node_width, widget.sizeHint().width() + 2 * self.title_padding)

    def get_selected_format(self):
        checked_button = self.output_format_group.checkedButton()
        if checked_button:
            return checked_button.text()
        raise ValueError("Must select an output format")
    
    def generate_output_path(self):
        store_dirs = {}
        for output_port in self.output_ports:
            if output_port.is_connected():
                for connected_port in output_port.connected_ports:
                    store_dirs[output_port.port_label] = []
                    parent_node = connected_port.parent_node
                    if isinstance(parent_node, OutputNode):
                        store_dirs[output_port.port_label].append(parent_node.output_path)
                    elif isinstance(parent_node, ModelNode):
                        store_dirs[output_port.port_label].append(
                            os.path.join(TEMP_PATH, f"model_node_{parent_node.index}"))
        return store_dirs

    def generate_input_path(self):
        if len(self.input_ports[0].connected_ports) == 1:
            input_path = self.input_ports[0].connected_ports[0].port_value
        else:
            raise ValueError("One model node should only have one input path.")
        return input_path
    
    def get_device(self):
        for bool_port in self.bool_ports:
            if bool_port.port_label == 'Use CPU':
                return 'cpu'
        else:
            return None


class MSSTModelNode(ModelNode):
    def __init__(self, model_class = None, model_name = None, model_type = None, input_ports = None, param_ports = None,
                 output_ports = None, bool_ports = None, scene = None, parent = None, upstream_node = None,
                 downstream_nodes = None):
        super().__init__(model_class, model_name, model_type, input_ports, param_ports, output_ports, bool_ports, scene,
                         parent, upstream_node, downstream_nodes)

    def init_ports(self):
        msst_model_path = os.path.join(current_dir, '..', '..', '..', 'data', 'msst_model_map.json')
        with open(msst_model_path, 'r') as f:
            model_map = json.load(f)
        for _ in model_map[self._model_class]:
            if _['name'] == self._model_name:
                self._config_path = _['config_path']
                self._model_type = _['model_type']

        with open(self._config_path) as f:
            if self._model_type == 'htdemucs':
                self._config = OmegaConf.load(self._config_path)
            else:
                self._config = ConfigDict(yaml.load(f, Loader = yaml.FullLoader))

        self.input_ports = [InputPort("Input")]
        for instrument in self._config.training.instruments:
            self.output_ports.append(OutputPort(instrument))
        for param in self._config.inference:
            if param != "normalize":
                self.param_ports.append(ParamPort(port_label = param, default_value = self._config.inference[param]))
        self.bool_ports = [BoolPort(port_label = "Use CPU", default_value = False)]
        if "normalize" in self._config.inference:
            self.bool_ports.append(
                BoolPort(port_label = "Normalize", default_value = self._config.inference["normalize"]))
        self.bool_ports.append(BoolPort(port_label = "Use TTA", default_value = False))

        # print('input_ports:', self.input_ports)
        # print('output_ports:', self.output_ports)
        # print('param_ports:', self.param_ports)
        # print('bool_ports:', self.bool_ports)

    def write_to_config(self):
        for param_port in self.param_ports:
            param = param_port.port_label
            self._config.inference[param] = param_port.port_value
        for bool_port in self.bool_ports:
            if bool_port.port_label == "Normalize":
                self._config.inference["normalize"] = bool_port.port_value

        with open(self._config_path, 'w') as f:
            yaml.dump(self._config.to_dict(), f)

    def run(self):
        self.write_to_config()
        store_dirs = self.generate_output_path()
        input_path = self.generate_input_path()

        use_tta, normalize = False, False
        for bool_port in self.bool_ports:
            if bool_port._port_label == "Normalize":
                normalize = bool_port.port_value
            if bool_port._port_label == "USE TTA":
                use_tta = bool_port.port_value

        msst_seperate = ComfyMSST(
            model_type = self._model_type,
            config_path = self._config_path,
            model_path = os.path.join(current_dir, '..', '..', '..', ".\\pretrain", self._model_class, self._model_name),
            device = self.get_device(),
            output_format = self.get_selected_format(),
            store_dirs = store_dirs,
            use_tta = use_tta,
            normalize = normalize,
        )

        msst_seperate.process_folder(input_folder = input_path)

    


class VRModelNode(ModelNode):
    def __init__(self, model_class = None, model_name = None, input_ports = None, param_ports = None,
                 output_ports = None, bool_ports = None, scene = None, parent = None, upstream_node = None,
                 downstream_nodes = None):
        super().__init__(model_class, model_name, input_ports, param_ports, output_ports, bool_ports, scene, parent,
                         upstream_node, downstream_nodes)

    def init_ports(self):
        vr_model_path = os.path.join(current_dir, '..', '..', '..', 'data', 'vr_model_map.json')
        with open(vr_model_path, 'r') as f:
            model_map = json.load(f)

        self.input_ports = [InputPort("Input")]
        self.output_ports.append(OutputPort(model_map[self._model_name]["primary_stem"]))
        self.output_ports.append(OutputPort(model_map[self._model_name]["secondary_stem"]))
        self.param_ports.append(ParamPort("Normalization Threshold", default_value = 0.9))
        self.param_ports.append(ParamPort("Batch Size", default_value = 4))
        self.param_ports.append(ParamPort("Window Size", default_value = 512))
        self.param_ports.append(ParamPort("Aggression", default_value = 5))
        self.param_ports.append(ParamPort("Post Process Threshold", default_value = 0.2))
        self.bool_ports.append(BoolPort("Use CPU", default_value = False))
        self.bool_ports.append(BoolPort("Invert Spect", default_value = False))
        self.bool_ports.append(BoolPort("Enable Tta", default_value = False))
        self.bool_ports.append(BoolPort("High End Process", default_value = False))
        self.bool_ports.append(BoolPort("Enable Post Process", default_value = False))

    def run(self):
        store_dirs = self.generate_output_path()
        input_path = self.generate_input_path()
        normalization_threshold = 0.9
        invert_using_spec = False
        use_cpu = False
        vr_params = {"batch_size": 2, "window_size": 512, "aggression": 5, "enable_tta": False,
                         "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},

        for param_port in self.param_ports:
            label = param_port.port_label
            if label == "Normalization Threshold":
                normalization_threshold = param_port.port_value

            elif label == "Batch Size":
                vr_params["batch_size"] = param_port.port_value
            elif label == "Aggression":
                vr_params["aggression"] = param_port.port_value
            elif label == "Post Process Threshold":
                vr_params["post_process_threshold"] = param_port.port_value
        
        for bool_port in self.bool_ports:
            label = bool_port.port_label
            
            if label == "Invert Spect":
                invert_using_spec = bool_port.port_value
            elif label == "Enable TTA":
                vr_params["enable_tta"] = bool_port.port_value
            elif label == "High End Process":
                vr_params["high_end_process"] = bool_port.port_value
            elif label == "Enable Post Process":
                vr_params["enable_post_process"] = bool_port.port_value
            elif label == "Use CPU":
                use_cpu = bool_port.port_value



        vr_separate = ComfyVR(
            model_file = os.path.join(current_dir, '..', '..', '..', 'pretrain', 'VR_Models', self._model_name),
            output_format = self.get_selected_format(),
            normalization_threshold = normalization_threshold,
            output_single_stem = None,
            invert_using_spec = invert_using_spec,
            use_cpu = use_cpu,
            vr_params = vr_params,
            store_dirs = self.generate_output_path()

        )

        vr_separate.separate(self.generate_input_path)
