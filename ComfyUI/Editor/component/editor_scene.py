import json
import math
import uuid
import os
from qfluentwidgets import InfoBar, InfoBarPosition
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView
from PySide6.QtGui import QBrush, QColor, QPen, QPainter
from PySide6.QtCore import QLine, Qt
from ComfyUI.Editor.component.edge import NodeEdge, DraggingEdge
from ComfyUI.Editor.component.node_port import InputPort, OutputPort
from ComfyUI.Editor.component.node import InputNode, OutputNode, FileInputNode, ModelNode
from ComfyUI.Editor.component.node_executor import NodeExecutor, NodeExecutorThread

class EditorScene(QGraphicsScene):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.view = None
        self.dragging_edge_mode = False
        self.input_node = None
        self.nodes = {}
        self.copy_data = {}
        self.nodes_to_run = []
        self.tmp_dir = "./tmp"

    def drawBackground(self, painter: QPainter, rect) -> None:
        self.setBackgroundBrush(QBrush(QColor('#212121')))
        self.width = 32000
        self.height = 32000
        self.grid_size = 20
        self.chunk_size = 10
        self.setSceneRect(-self.width / 2, -self.height / 2, self.width, self.height)
        self.setItemIndexMethod(QGraphicsScene.BspTreeIndex)

        self.normal_line_pen = QPen(QColor("#313131"))
        self.normal_line_pen.setWidthF(1.0)
        self.dark_line_pen = QPen(QColor("#151515"))
        self.dark_line_pen.setWidthF(1.5)

        super().drawBackground(painter, rect)
        lines, drak_lines = self.cal_grid_lines(rect)
        painter.setPen(self.normal_line_pen)
        painter.drawLines(lines)

        painter.setPen(self.dark_line_pen)
        painter.drawLines(drak_lines)

    def cal_grid_lines(self, rect):
        left, right, top, bottom = math.floor(rect.left()), math.floor(
            rect.right()), math.floor(rect.top()), math.floor(rect.bottom())

        first_left = left - (left % self.grid_size)
        first_top = top - (top % self.grid_size)

        lines = []
        drak_lines = []
        for v in range(first_top, bottom, self.grid_size):
            line = QLine(left, v, right, v)
            if v % (self.grid_size * self.chunk_size) == 0:
                drak_lines.append(line)
            else:
                lines.append(line)

        for h in range(first_left, right, self.grid_size):
            line = QLine(h, top, h, bottom)
            if h % (self.grid_size * self.chunk_size) == 0:
                drak_lines.append(line)
            else:
                lines.append(line)

        return lines, drak_lines
    
    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        
        # 检查是否点击了一个端口 (InputPort 或 OutputPort)
        if isinstance(item, InputPort) or isinstance(item, OutputPort):
            self.dragging_edge_mode = True
            self.source_port = item
            self.dragging_edge = DraggingEdge(scene=self)  # 创建拖拽连接线
            self.dragging_edge.setSourcePos(self.source_port.getPortPos())  # 设置起始位置
            self.addItem(self.dragging_edge)  # 将拖拽连接线添加到场景中
            self.views()[0].setDragMode(QGraphicsView.NoDrag)
        else:    
            super().mousePressEvent(event)

    # 处理鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.dragging_edge_mode:
            self.dragging_edge.updatePathDuringDrag(event.scenePos())  # 更新拖拽路径
        super().mouseMoveEvent(event)

    # 处理鼠标释放事件
    def mouseReleaseEvent(self, event):
        if self.dragging_edge_mode:
            item = self.itemAt(event.scenePos(), self.views()[0].transform())

            if isinstance(item, InputPort) or isinstance(item, OutputPort):
                if self.source_port != item:
                    self.createNodeEdge(self.source_port, item)

            self.removeItem(self.dragging_edge)
            self.dragging_edge = None
            self.source_port = None
            self.dragging_edge_mode = False
            self.views()[0].setDragMode(QGraphicsView.RubberBandDrag)

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.removeSelectedItems()
            
        elif event.key() == Qt.Key_A and event.modifiers() == Qt.ControlModifier:
            self.selectAll()
            
        elif event.key() == Qt.Key_C and event.modifiers() == Qt.ControlModifier:
            self.copySelectedItems()   
            
        elif event.key() == Qt.Key_V and event.modifiers() == Qt.ControlModifier:
            self.pasteItems(self.views()[0].mapToScene(self.views()[0].mapFromGlobal(self.views()[0].cursor().pos())))
            
        super().keyPressEvent(event)       

    def addNodeFromText(self, text, pos):
        if text == "Input Node":
            node = InputNode(path="input/")
        elif text == "Output Node":
            node = OutputNode(path="output/")
        elif text == "File Input Node":
            node = FileInputNode(path="tmp/input/")
        else:
            node = ModelNode(text)
        node.setPos(pos)
        self.addNode(node)

    def createNodeEdge(self, port1, port2, from_load=False):
        if isinstance(port1, InputPort) and isinstance(port2, OutputPort):
            upper_port = port2
            lower_port = port1
            
        elif isinstance(port1, OutputPort) and isinstance(port2, InputPort):
            upper_port = port1
            lower_port = port2
        else:
            return    
        upper_node = upper_port.parent_node
        lower_node = lower_port.parent_node

        node_edge = NodeEdge(upper_port, lower_port, scene=self)
        
        if not from_load:
            upper_node.addDownStreamNode(lower_node.node_dict["uid"], upper_port)
            lower_node.addUpStreamNode(upper_node.node_dict["uid"])

        upper_node.edges.append(node_edge)
        lower_node.edges.append(node_edge)
        upper_port.connected_edges.append(node_edge)
        lower_port.connected_edges.append(node_edge)
        self.addItem(node_edge)
        upper_port.updateConnectionState()
        lower_port.updateConnectionState()

    def removeNodeEdge(self, edge):
        upper_port = edge.upper_port
        lower_port = edge.lower_port
        upper_node = upper_port.parent_node
        lower_node = lower_port.parent_node
        upper_node.removeDownStreamNode(lower_node.node_dict["uid"], upper_port)
        lower_node.removeUpStreamNode(upper_node.node_dict["uid"])
        upper_node.edges.remove(edge)
        lower_node.edges.remove(edge)
        upper_port.connected_edges.remove(edge)
        lower_port.connected_edges.remove(edge)
        self.removeItem(edge)
        upper_port.updateConnectionState()
        lower_port.updateConnectionState()

    def addNode(self, node, uid=None):
        if isinstance(node, InputNode) or isinstance(node, FileInputNode):
            if self.input_node:
                InfoBar.error(
                    title='Error',
                    content="Only one Input Node is allowed",
                    orient=Qt.Vertical,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=-1,
                    parent=self.views()[0]
                )
                return
            self.input_node = node

        self.addItem(node)
        if uid is not None:
            node.node_dict["uid"] = uid
        else:
            node.node_dict["uid"] = str(uuid.uuid4())
        self.nodes[node.node_dict["uid"]] = node

    def removeNode(self, node):
        self.removeItem(node)
        self.nodes.pop(node.node_dict["uid"])
        edges_to_remove = node.edges.copy()
        for edge in edges_to_remove:
            self.removeNodeEdge(edge)
        if isinstance(node, InputNode) or isinstance(node, FileInputNode):
            self.input_node = None

    def saveToJson(self, save_path):
        data = {}
        for uid in self.nodes:
            node = self.nodes[uid]
            node.node_dict["scene_pos"] = [node.scenePos().x(), node.scenePos().y()]
            # print(node.node_dict)
            data[node.node_dict["uid"]] = node.node_dict
        with open(save_path, 'w') as file:
            json.dump(data, file, indent=4)

    def loadFromJson(self, load_path):
        data = json.load(open(load_path, 'r'))
        for index in data:
            node_data = data[index]
            if node_data["model_name"] == "Input Node":
                node = InputNode(path=node_data["parameter"][0]["current_value"])
            elif node_data["model_name"] == "Output Node":
                node = OutputNode(path=node_data["parameter"][0]["current_value"])
            elif node_data["model_name"] == "File Input Node":
                node = FileInputNode(path=node_data["parameter"][0]["current_value"])
            else:
                node = ModelNode(node_data["model_name"])

            node.node_dict = node_data
            # print(node.node_dict)
            node.setPos(node_data["scene_pos"][0], node_data["scene_pos"][1])
            self.addNode(node, uid=node_data["uid"])

        for index in data:
            node_data = data[index]
            if node_data["down_stream_nodes"]:
                for item in node_data["down_stream_nodes"]:
                    downstream_node_uid, port_index = item
                    # print(index, downstream_node_index, port_index)
                    upper_node = self.nodes[node_data["uid"]]
                    lower_node = self.nodes[downstream_node_uid]
                    upper_port = upper_node.output_ports[port_index]
                    lower_port = lower_node.input_ports[0]
                    self.createNodeEdge(upper_port, lower_port, from_load=True)
                    
    def removeSelectedItems(self):
        for item in self.selectedItems():
                if not isinstance(item, NodeEdge):
                    self.removeNode(item)
                    
    def selectAll(self):
        for item in self.items():
            item.setSelected(True)
            
    def copySelectedItems(self):
        self.copy_data = {
            "nodes": [],
            "edges": []
        }
        node_map = {}  # 用于记录旧节点与新节点的映射

        for item in self.selectedItems():
            if not isinstance(item, NodeEdge):
                node_data = {
                    "uid": item.node_dict["uid"],
                    "model_name": item.node_dict["model_name"],
                    "parameters": item.node_dict["parameter"],
                    "pos": item.pos(),
                    "down_stream_nodes": item.node_dict.get("down_stream_nodes", [])
                }
                self.copy_data["nodes"].append(node_data)
                node_map[item.node_dict["uid"]] = node_data

        # 记录边信息
        for node_data in self.copy_data["nodes"]:
            for downstream in node_data["down_stream_nodes"]:
                downstream_uid, output_port_index = downstream
                if downstream_uid in node_map:
                    self.copy_data["edges"].append((node_data["uid"], downstream_uid, output_port_index))

        # print("Copied data:", self.copy_data)
            
    def pasteItems(self, new_pos):
        if not self.copy_data["nodes"]:
            return

        uid_map = {}  # 用于替换旧的 UID 为新的 UID
        # 计算第一个节点的位置偏移量
        offset = new_pos - self.copy_data["nodes"][0]["pos"]

        new_nodes = {}
        # 创建节点
        for node_data in self.copy_data["nodes"]:
            if node_data["model_name"] == "Input Node":
                node = InputNode(path=node_data["parameters"][0]["current_value"])
            elif node_data["model_name"] == "Output Node":
                node = OutputNode(path=node_data["parameters"][0]["current_value"])
            elif node_data["model_name"] == "File Input Node":
                node = FileInputNode(path=node_data["parameters"][0]["current_value"])
            else:
                node = ModelNode(node_data["model_name"])

            new_uid = str(uuid.uuid4())
            uid_map[node_data["uid"]] = new_uid
            node.node_dict["uid"] = new_uid
            node.setPos(node_data["pos"] + offset)
            self.addNode(node)
            new_nodes[new_uid] = node

        # 创建新的edge
        for source_uid, target_uid, output_port_index in self.copy_data["edges"]:
            if source_uid in uid_map and target_uid in uid_map:
                upper_node = new_nodes[uid_map[source_uid]]
                lower_node = new_nodes[uid_map[target_uid]]
                # 根据记录的 output_port_index 获取正确的输出端口
                if hasattr(upper_node, 'output_ports') and len(upper_node.output_ports) > output_port_index:
                    upper_port = upper_node.output_ports[output_port_index]
                    lower_port = lower_node.input_ports[0] # 输入端口有且仅有一个
                    self.createNodeEdge(upper_port, lower_port)

    def clearItems(self):
        self.selectAll()
        self.removeSelectedItems()

    def generatePath(self, node):
        
        if isinstance(node, ModelNode):
            self.nodes_to_run.append(node)
            upstream_node_uid = node.node_dict["up_stream_node"]
            upstream_node = self.nodes[upstream_node_uid]
            if isinstance(upstream_node, InputNode):
                input_path = upstream_node.node_dict["parameter"][0]["current_value"]
                node.node_dict["input_path"] = input_path
            else:
                pass
            output_path_dict = {}        
            for item in node.node_dict["down_stream_nodes"]:
                downstream_node_uid, port_index = item
                downstream_node = self.nodes[downstream_node_uid]
                instrument = node.node_dict["output"][port_index]
                if isinstance(downstream_node, OutputNode):
                    output_path = downstream_node.node_dict["parameter"][0]["current_value"]
                    output_path_dict[instrument] = output_path
                else:
                    output_path = os.path.join(self.tmp_dir, f"{node.node_dict['uid']}_{instrument}")
                    os.makedirs(output_path, exist_ok=True)
                    output_path_dict[instrument] = output_path
                    downstream_node.node_dict["input_path"] = output_path
            node.node_dict["output_path"] = output_path_dict        

        for item in node.node_dict["down_stream_nodes"]:
            downstream_node_uid, port_index = item
            downstream_node = self.nodes[downstream_node_uid]
            self.generatePath(downstream_node)

    def run(self):
        if self.input_node is None:
            InfoBar.error(
                title='Error',
                content="No Input Node",
                orient=Qt.Vertical,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=-1,
                parent=self.views()[0]
            )
            return
        self.generatePath(self.input_node) 
        
        