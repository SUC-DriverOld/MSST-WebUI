import json
import math
from qfluentwidgets import InfoBar, InfoBarPosition
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView
from PySide6.QtGui import QBrush, QColor, QPen, QPainter
from PySide6.QtCore import QLine, Qt
from ComfyUI.Editor.component.edge import NodeEdge, DraggingEdge
from ComfyUI.Editor.component.node_port import InputPort, OutputPort
from ComfyUI.Editor.component.node import InputNode, OutputNode, FileInputNode, ModelNode

class EditorScene(QGraphicsScene):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.view = None
        self.dragging_edge_mode = False
        self.input_node = None
        self.nodes = []

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
            for item in self.selectedItems():
                if isinstance(item, NodeEdge):
                    self.removeNodeEdge(item)
                else:
                    self.removeNode(item)
            super().keyPressEvent(event)

    def addNodeFromText(self, text, pos):
        if text == "Input Node":
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
            node = InputNode(path="input/")
            self.input_node = node
        elif text == "Output Node":
            node = OutputNode(path="output/")
        elif text == "File Input Node":
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
            node = FileInputNode(path="tmp/input/")
            self.input_node = node
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
            upper_node.addDownStreamNode(lower_node.node_dict["index"], upper_port)
            lower_node.addUpStreamNode(upper_node.node_dict["index"])

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
        upper_node.removeDownStreamNode(lower_node.node_dict["index"], upper_port)
        lower_node.removeUpStreamNode(upper_node.node_dict["index"])
        upper_node.edges.remove(edge)
        lower_node.edges.remove(edge)
        upper_port.connected_edges.remove(edge)
        lower_port.connected_edges.remove(edge)
        self.removeItem(edge)
        upper_port.updateConnectionState()
        lower_port.updateConnectionState()

    def addNode(self, node, index=None):
        self.addItem(node)
        if index is not None:
            node.node_dict["index"] = index
        else:
            node.node_dict["index"] = len(self.nodes)
        self.nodes.append(node)

    def removeNode(self, node):
        self.removeItem(node)
        self.nodes.remove(node)
        edges_to_remove = node.edges.copy()
        for edge in edges_to_remove:
            self.removeNodeEdge(edge)
        if isinstance(node, InputNode) or isinstance(node, FileInputNode):
            self.input_node = None

    def saveToJson(self, save_path):
        data = {}
        for node in self.nodes:
            node.node_dict["scene_pos"] = [node.scenePos().x(), node.scenePos().y()]
            # print(node.node_dict)
            data[node.node_dict["index"]] = node.node_dict
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
            self.addNode(node, index)

        for index in data:
            node_data = data[index]
            if node_data["down_stream_nodes"]:
                for item in node_data["down_stream_nodes"]:
                    downstream_node_index, port_index = item
                    # print(index, downstream_node_index, port_index)
                    upper_node = self.nodes[int(index)]
                    lower_node = self.nodes[int(downstream_node_index)]
                    upper_port = upper_node.output_ports[port_index]
                    lower_port = lower_node.input_ports[0]
                    self.createNodeEdge(upper_port, lower_port, from_load=True)