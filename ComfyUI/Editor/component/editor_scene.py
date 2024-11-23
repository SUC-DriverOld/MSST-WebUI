import sys
import math
from PySide6.QtWidgets import QGraphicsScene
from PySide6.QtGui import QBrush, QColor, QPen, QPainter
from PySide6.QtCore import QLine
from ComfyUI.Editor.component.edge import NodeEdge, DraggingEdge
from ComfyUI.Editor.component.node_port import InputPort, OutputPort

class EditorScene(QGraphicsScene):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.view = None
        self.dragging_edge_mode = False

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
            print(item)

            if isinstance(item, InputPort) or isinstance(item, OutputPort):
                if self.source_port != item:
                    node_edge = NodeEdge(self.source_port, item, scene=self)
                    self.addItem(node_edge)

            self.removeItem(self.dragging_edge)
            self.dragging_edge = None
            self.source_port = None
            self.dragging_edge_mode = False

        super().mouseReleaseEvent(event)

    def createNodeEdge(self, source_port, des_port):
        if isinstance(source_port, InputPort) and isinstance(des_port, OutputPort):
            upper_port = des_port
            lower_port = source_port
            upper_node = des_port.parent_node
            lower_node = source_port.parent_node
        elif isinstance(source_port, OutputPort) and isinstance(des_port, InputPort):
            upper_port = source_port
            lower_port = des_port
            upper_node = source_port.parent_node
            lower_node = des_port.parent_node
        else:
            return    

        node_edge = NodeEdge(source_port, des_port, scene=self)

        
        self.addItem(node_edge)
        return node_edge    