"""
example of a node_dict:
{   
    "uid": None # unique id of the node, default is None
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
    "down_stream_nodes": [["example-uid1", 0], ["example-uid2", 1]], # list of downstream nodes, each element is a list of [uid, output_port_index], default is []
    "up_stream_node": "example-uid3", # uid of the upstream node, default is None, note that there is only one upstream node
    "output_format": "wav", # output format of the node, in ["wav", "mp3", "flac"]
    "scene_pos": [0, 0], # position of the node in the scene, default is [0, 0]
    "input_path": None, # path of the input folder, default is None
    "output_path": None, # path of the output folder, default is None. Note that str for single folder, dict with instrument keys for multiple folders
    # example of an output path:
    # {
    #     "vocal": "./output/vocal",
    #     "instruments": "./output/instruments"
    # }
}
"""

# import sys
# sys.path.append("D:\projects\python\MSST-WebUI")

from PySide6.QtWidgets import QGraphicsItem, QGraphicsDropShadowEffect
from PySide6.QtGui import QFont, QPen, QFontMetrics, QBrush, QColor, QPainterPath
from PySide6.QtCore import Qt, QRectF
import json
from ComfyUI.Editor.component.node_port import InputPort, OutputPort, ParameterPort, BoolPort, FormatSelector
from ComfyUI.Editor.component.file_drag_area import FileDragArea
from ComfyUI.Editor.common.config import color, font


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
        self.edges = []
        self.add_ports()
        self.init_shadow()
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        
    def init_shadow(self):
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(20)
        self.shadow_color = QColor('#aaeeee00')

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

        self.shadow.setColor('#00000000')
        self.setGraphicsEffect(self.shadow)

        if self.isSelected():
            self.shadow.setColor(self.shadow_color)
            self.setGraphicsEffect(self.shadow)

        node_outline = QPainterPath()
        node_outline.addRoundedRect(0, 0, self.width, self.height, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor('#dd151515')))
        painter.drawPath(node_outline.simplified())

        title_outline = QPainterPath()
        title_outline.setFillRule(Qt.WindingFill)
        title_outline.addRoundedRect(0, 0, self.width, 30, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPath(title_outline.simplified())
        
        self.label = self.model_name
        painter.setFont(font)
        painter.setPen(QPen(Qt.white))
        painter.setBrush(color)
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.label)
        max_text_width = 190
        if text_width > max_text_width:
            truncated_text = font_metrics.elidedText(self.label, Qt.ElideRight, max_text_width)
        else:
            truncated_text = self.label
        text_height = font_metrics.height()
        text_rect = QRectF(5, (30 - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)
        
    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def add_ports(self):
        for i, text in enumerate(self.node_dict["input"]):
            port = InputPort(text=text)
            port.setParentItem(self)
            port.setPos(0, 30 + i * 20)
            port.setParentNode(self, index=i)
            self.input_ports.append(port)
        for i, text in enumerate(self.node_dict["output"]):
            port = OutputPort(text=text)
            port.setParentItem(self)
            port.setPos(100, 30 + i * 20)
            port.setParentNode(self, index=i)
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
            port.setPos(0, 30 + (max(self.len[0], self.len[1]) + i) * 20)
            port.setParentNode(self, index=i)
            self.parameter_ports.append(port)
            
        for i, param in enumerate(self.node_dict["bool"]):
            port = BoolPort(
                parameter=param["parameter"],
                default_value=param["default_value"],
                current_value=param["current_value"]
            )
            port.setParentItem(self)
            port.setPos(0, 30 + (max(self.len[0], self.len[1]) + self.len[2] + i) * 20)
            port.setParentNode(self, index=i)
            self.bool_ports.append(port)
            
        self.format_selector = FormatSelector()
        for i, format in enumerate(["wav", "mp3", "flac"]):
            if format == self.node_dict["output_format"]:
                self.format_selector.select(i)
                break
        self.format_selector.setParentItem(self)
        self.format_selector.setParentNode(self)
        self.format_selector.setPos(0, 30 + (max(self.len[0], self.len[1]) + self.len[2] + self.len[3]) * 20)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # 如果节点位置变化，则需要更新端口位置
            self.updatePortPositions()
        return super().itemChange(change, value)    
    
    def updatePortPositions(self):
        for edge in self.edges:
            edge.updatePath()

    def addDownStreamNode(self, target_node_uid, output_port):
        for i, port in enumerate(self.output_ports):
            if port == output_port:
                self.node_dict["down_stream_nodes"].append([target_node_uid, i])
                break
        # print(self.node_dict["down_stream_nodes"])

    def removeDownStreamNode(self, target_node_uid, output_port):
        for i, port in enumerate(self.output_ports):
            if port == output_port:
                try:
                    self.node_dict["down_stream_nodes"].remove([target_node_uid, i])
                    break
                except:
                    print(f"{self.node_dict['down_stream_nodes']}, {target_node_uid}, {output_port}, {i}")
        # print(self.node_dict["down_stream_nodes"])

    def addUpStreamNode(self, target_node_uid):
        if self.node_dict["up_stream_node"] is None:
            self.node_dict["up_stream_node"] = target_node_uid
        else:
            print("There is already an upstream node.")

    def removeUpStreamNode(self, target_node_uid):
        if self.node_dict["up_stream_node"] == target_node_uid:
            self.node_dict["up_stream_node"] = None
        else:
            print("The target node is not the upstream node.")


class InputNode(QGraphicsItem):
    def __init__(self, parent = None, path = "input/"):
        super().__init__(parent)
        self.node_dict = {
            "uid": None,
            "model_name": "Input Node",
            "parameter": [
                {
                    "parameter": "folder_path",
                    "type": "str",
                    "default_value": f"{path}",
                    "max_value": None,
                    "min_value": None,
                    "current_value": f"{path}"
                }
            ],
            "down_stream_nodes": [],
            "up_stream_node": None,
            "scene_pos": [0, 0]
        }
        self.width = 200
        self.height = 70
        self.edges = []
        
        self.init_shadow()
        self.add_ports()
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)

    def init_shadow(self):
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(20)
        self.shadow_color = QColor('#aaeeee00')

    def paint(self, painter, option, widget):
        self.shadow.setColor('#00000000')
        self.setGraphicsEffect(self.shadow)

        if self.isSelected():
            self.shadow.setColor(self.shadow_color)
            self.setGraphicsEffect(self.shadow)

        node_outline = QPainterPath()
        node_outline.addRoundedRect(0, 0, self.width, self.height, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor('#dd151515')))
        painter.drawPath(node_outline.simplified())

        title_outline = QPainterPath()
        title_outline.setFillRule(Qt.WindingFill)
        title_outline.addRoundedRect(0, 0, self.width, 30, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPath(title_outline.simplified())
        
        self.label = "Input Node"
        painter.setFont(font)
        painter.setPen(QPen(Qt.white))
        painter.setBrush(color)
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.label)
        max_text_width = 190
        if text_width > max_text_width:
            truncated_text = font_metrics.elidedText(self.label, Qt.ElideRight, max_text_width)
        else:
            truncated_text = self.label
        text_height = font_metrics.height()
        text_rect = QRectF(5, (30 - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def add_ports(self):
        self.output_port = OutputPort(text="Input")
        self.output_port.setParentItem(self)
        self.output_port.setPos(100, 30)
        self.output_port.setParentNode(self, index=0)
        self.output_ports = [self.output_port]

        self.parameter_port = ParameterPort(
            parameter="folder_path",
            type=str,
            default_value=self.node_dict["parameter"][0]["default_value"],
            max_value=None,
            min_value=None,
            current_value=self.node_dict["parameter"][0]["default_value"]
        )

        self.parameter_port.setParentItem(self)
        self.parameter_port.setPos(0, 50)
        self.parameter_port.setParentNode(self, index=0)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # 如果节点位置变化，则需要更新端口位置
            self.updatePortPositions()
        return super().itemChange(change, value)    
    
    def updatePortPositions(self):
        for edge in self.edges:
            edge.updatePath() 

    def addDownStreamNode(self, target_node_uid, output_port):
        if output_port == self.output_port:
            self.node_dict["down_stream_nodes"].append([target_node_uid, 0])

    def removeDownStreamNode(self, target_node_uid, output_port):
        if output_port == self.output_port:
            try:
                self.node_dict["down_stream_nodes"].remove([target_node_uid, 0])
            except:
                print(f"{self.node_dict['down_stream_nodes']}, {target_node_uid}, {output_port}")


        
class OutputNode(QGraphicsItem):
    def __init__(self, parent = None, path = "output/"):
        super().__init__(parent)
        self.node_dict = {
            "uid": None,
            "model_name": "Output Node",
            "parameter": [
                {
                    "parameter": "folder_path",
                    "type": "str",
                    "default_value": f"{path}",
                    "max_value": None,
                    "min_value": None,
                    "current_value": f"{path}"
                }
            ],
            "down_stream_nodes": [],
            "up_stream_node": None,
            "scene_pos": [0, 0]

        }
        self.width = 200
        self.height = 70
        self.edges = []
        self.init_shadow()
        self.add_ports()
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)

    def init_shadow(self):
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(20)
        self.shadow_color = QColor('#aaeeee00')

    def paint(self, painter, option, widget):
        self.shadow.setColor('#00000000')
        self.setGraphicsEffect(self.shadow)

        if self.isSelected():
            self.shadow.setColor(self.shadow_color)
            self.setGraphicsEffect(self.shadow)

        node_outline = QPainterPath()
        node_outline.addRoundedRect(0, 0, self.width, self.height, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor('#dd151515')))
        painter.drawPath(node_outline.simplified())

        title_outline = QPainterPath()
        title_outline.setFillRule(Qt.WindingFill)
        title_outline.addRoundedRect(0, 0, self.width, 30, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPath(title_outline.simplified())
        
        self.label = "Output Node"
        painter.setFont(font)
        painter.setPen(QPen(Qt.white))
        painter.setBrush(color)
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.label)
        max_text_width = 190
        if text_width > max_text_width:
            truncated_text = font_metrics.elidedText(self.label, Qt.ElideRight, max_text_width)
        else:
            truncated_text = self.label
        text_height = font_metrics.height()
        text_rect = QRectF(5, (30 - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def add_ports(self):
        self.input_port = InputPort(text="Output")
        self.input_port.setParentItem(self)
        self.input_port.setPos(0, 30)
        self.input_port.setParentNode(self, index=0)
        self.input_ports = [self.input_port]

        self.parameter_port = ParameterPort(
            parameter="folder_path",
            type=str,
            default_value=self.node_dict["parameter"][0]["default_value"],
            max_value=None,
            min_value=None,
            current_value=self.node_dict["parameter"][0]["default_value"]
        )

        self.parameter_port.setParentItem(self)
        self.parameter_port.setPos(0, 50)
        self.parameter_port.setParentNode(self, index=0)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # 如果节点位置变化，则需要更新端口位置
            self.updatePortPositions()
        return super().itemChange(change, value)    
    
    def updatePortPositions(self):
        for edge in self.edges:
            edge.updatePath()  

    def addUpStreamNode(self, target_node_uid):
        if self.node_dict["up_stream_node"] is None:
            self.node_dict["up_stream_node"] = target_node_uid
        else:
            print("There is already an upstream node.")

    def removeUpStreamNode(self, target_node_uid):
        if self.node_dict["up_stream_node"] == target_node_uid:
            self.node_dict["up_stream_node"] = None
        else:
            print("The target node is not the upstream node.")        
            


class FileInputNode(QGraphicsItem):
    def __init__(self, parent = None, path = "input/"):
        super().__init__(parent)
        self.node_dict = {
            "uid": None,
            "model_name": "File Input Node",
            "parameter": [
                {
                    "parameter": "file_path",
                    "type": "str",
                    "default_value": f"{path}",
                    "max_value": None,
                    "min_value": None,
                    "current_value": f"{path}"
                }
            ],
            "down_stream_nodes": [],
            "up_stream_node": None,
            "scene_pos": [0, 0]
        }
        self.width = 200
        self.height = 200
        self.edges = []
        
        self.init_shadow()
        self.add_ports()
        self.add_file_drag_area()
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)   

    def init_shadow(self):
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(20)
        self.shadow_color = QColor('#aaeeee00')

    def paint(self, painter, option, widget):
        self.shadow.setColor('#00000000')
        self.setGraphicsEffect(self.shadow)

        if self.isSelected():
            self.shadow.setColor(self.shadow_color)
            self.setGraphicsEffect(self.shadow)

        node_outline = QPainterPath()
        node_outline.addRoundedRect(0, 0, self.width, self.height, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor('#dd151515')))
        painter.drawPath(node_outline.simplified())

        title_outline = QPainterPath()
        title_outline.setFillRule(Qt.WindingFill)
        title_outline.addRoundedRect(0, 0, self.width, 30, 5, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPath(title_outline.simplified())
        
        self.label = "File Input Node"
        painter.setFont(font)
        painter.setPen(QPen(Qt.white))
        painter.setBrush(color)
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.label)
        max_text_width = 190
        if text_width > max_text_width:
            truncated_text = font_metrics.elidedText(self.label, Qt.ElideRight, max_text_width)
        else:
            truncated_text = self.label
        text_height = font_metrics.height()
        text_rect = QRectF(5, (30 - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def add_ports(self):
        self.output_port = OutputPort(text="Input")
        self.output_port.setParentItem(self)
        self.output_port.setPos(100, 30)
        self.output_port.setParentNode(self, index=0)
        self.output_ports = [self.output_port]

    def add_file_drag_area(self):
        self.file_drag_area = FileDragArea(path=self.node_dict["parameter"][0]["default_value"])
        self.file_drag_area.setParentItem(self)
        self.file_drag_area.setPos(0, 50)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # 如果节点位置变化，则需要更新端口位置
            self.updatePortPositions()
        return super().itemChange(change, value)    
    
    def updatePortPositions(self):
        for edge in self.edges:
            edge.updatePath()

    def addDownStreamNode(self, target_node_uid, output_port):
        if output_port == self.output_port:
            self.node_dict["down_stream_nodes"].append([target_node_uid, 0])

    def removeDownStreamNode(self, target_node_uid, output_port):
        if output_port == self.output_port:
            self.node_dict["down_stream_nodes"].remove([target_node_uid, 0])
        

if __name__ == "__main__":
    from PySide6.QtWidgets import QGraphicsView, QApplication, QWidget, QVBoxLayout
    from PySide6.QtGui import QPainter
    app = QApplication(sys.argv)

    widget = QWidget()
    from ComfyUI.Editor.component.editor_scene import EditorScene
    scene = EditorScene()
    view = QGraphicsView(scene)
    widget.setFixedSize(1000, 500)

    test_node1 = ModelNode("7_HP2-UVR.pth")
    scene.addNode(test_node1)
    test_node1.setPos(50, 50)

    test_node2 = ModelNode("Apollo_LQ_MP3_restoration.ckpt")
    scene.addNode(test_node2)
    test_node2.setPos(300, 50)

    test_node3 = InputNode(path="input/")
    scene.addNode(test_node3)
    test_node3.setPos(550, 50)

    test_node4 = OutputNode(path="output/")
    scene.addNode(test_node4)
    test_node4.setPos(800, 50)

    test_node5 = FileInputNode(path="tmp/input/")
    scene.addNode(test_node5)
    test_node5.setPos(300, 300)

    
    view.setRenderHint(QPainter.Antialiasing)
    layout = QVBoxLayout()
    layout.addWidget(view)
    widget.setLayout(layout)
    widget.show()
    sys.exit(app.exec())