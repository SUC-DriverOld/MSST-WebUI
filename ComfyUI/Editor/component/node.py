"""
example of a node_dict:
{   
    "index": 0, # index of the node, default is -1
    "model_name": "model_bs_roformer_ep_317_sdr_12.9755.ckpt", # name of the model
    "model_type": "bs_roformer", # type of the model
    "path": "./pretrain/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "config_path: "./configs/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.yaml"
    "input: ["input"], # list of input ports
    "output": ["vocal", "instruments"], # list of output ports
    "parameter": [
        {
            "parameter": "batch_size",
            "type": "int",
            "default_value": 1,
            "max_value": 100,
            "min_value": 1,
            "current_value": 1
        },
        {
            "parameter": "dim_t",
            "type": "int",
            "default_value": 801,
            "max_value": 10000,
            "min_value": 1,
            "current_value": 801
        }
    ],
    "bool": [
        {
            "parameter": "Use TTA",
            "default_value": False,
            "current_value": False
        }
    ],
    down_stream_nodes: [[1, 0], [2, 1]], # list of downstream nodes, each element is a list of [index, output_port_index], default is []
    output_format: "wav", # output format of the node, in ["wav", "mp3", "flac"]
    scene_pos: [0, 0], # position of the node in the scene, default is [0, 0]
}
"""

from PySide6.QtWidgets import QGraphicsItem
from PySide6.QtGui import QFont, QPen, QFontMetrics
from PySide6.QtCore import Qt, QRectF
import json
from ComfyUI.Editor.component.node_port import InputPort, OutputPort, ParameterPort, BoolPort, FormatSelector
from ComfyUI.Editor.common.config import cfg
color = cfg.get(cfg.themeColor)
font = QFont("Consolas", 12)

class ModelNode(QGraphicsItem):
    def __init__(self, model_name, parent = None):
        super().__init__(parent)
        self.model_name = model_name
        self.node_dict = {}
        self.load_json(model_name)
        self.len = [len(self.node_dict["input"]), len(self.node_dict["output"]), len(self.node_dict["parameter"]), len(self.node_dict["bool"])]
        self.height = 30 + 20 * (max(self.len[0], self.len[1]) + self.len[2] + self.len[3] + 1)
        self.width = 200
        self.input_ports = []
        self.output_ports = []
        self.parameter_ports = []
        self.bool_ports = []
        self.add_ports()
        
        
    def load_json(self, model_name):
        path = f"ComfyUI/Editor/data/nodes/{model_name}.json"
        try:
            with open(path, "r") as f:
                self.node_dict = json.load(f)
        except Exception as e:
            raise e
        
    def save_json(self):
        path = f"ComfyUI/Editor/data/nodes/{self.model_name}.json"
        try:
            with open(path, "w") as f:
                json.dump(self.node_dict, f, indent=4)
        except Exception as e:
            raise e    
        
    def paint(self, painter, option, widget):
        self.label = self.model_name
        painter.setFont(font)
        painter.setPen(QPen(Qt.white))
        painter.setBrush(color)
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.label)
        max_text_width = 195
        if text_width > max_text_width:
            truncated_text = font_metrics.elidedText(self.label, Qt.ElideRight, max_text_width)
        else:
            truncated_text = self.label
        text_height = font_metrics.height()
        text_rect = QRectF(2.5, (self.height - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)
        
    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def add_ports(self):
        for i, text in enumerate(self.node_dict["input"]):
            port = InputPort(text=text)
            port.setParentItem(self)
            port.setPos(0, 30 + i * 20)
            self.input_ports.append(port)
        for i, text in enumerate(self.node_dict["output"]):
            port = OutputPort(text=text)
            port.setParentItem(self)
            port.setPos(100, 30 + i * 20)
            self.output_ports.append(port)
        for i, param in enumerate(self.node_dict["parameter"]):
            port = ParameterPort(
                parameter=param["parameter"],
                type=int if param["type"] == "int" else float,
                default_value=param["default_value"],
                max_value=param["max_value"],
                min_value=param["min_value"],
                current_value=param["current_value"]
            )
            port.setParentItem(self)
            port.setPos(0, 30 + (max(self.len[0] + self.len[1]) + i) * 20)
            self.parameter_ports.append(port)
            
        for i, param in enumerate(self.node_dict["bool"]):
            port = BoolPort(
                parameter=param["parameter"],
                default_value=param["default_value"],
                current_value=param["current_value"]
            )
            port.setParentItem(self)
            port.setPos(0, 30 + (max(self.len[0] + self.len[1]) + self.len[2] + i) * 20)
            self.bool_ports.append(port)
            
        format_selector = FormatSelector()
        