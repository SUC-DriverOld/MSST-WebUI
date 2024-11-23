from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsItem
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainterPath, QPen, QColor
import sys
sys.path.append("D:\projects\python\MSST-WebUI")
from ComfyUI.Editor.common.config import color
from ComfyUI.Editor.component.node_port import InputPort, OutputPort



class DraggingEdge(QGraphicsPathItem):
    def __init__(self, parent=None, scene=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setZValue(-1)
        self.source_pos = None
        self.des_pos = None

    def setSourcePos(self, pos: QPointF):
        self.source_pos = pos
        self.updatePath()

    def updatePathDuringDrag(self, current_pos):
        if self.source_pos:
            self.setDesPos(current_pos)

    def setDesPos(self, pos: QPointF):
        self.des_pos = pos
        self.updatePath()

    def updatePath(self):
        if self.source_pos and self.des_pos:
            control_point1 = QPointF((self.source_pos.x() + self.des_pos.x()) / 2, self.source_pos.y())
            control_point2 = QPointF((self.source_pos.x() + self.des_pos.x()) / 2, self.des_pos.y())

            path = QPainterPath(self.source_pos)
            path.cubicTo(control_point1, control_point2, self.des_pos)
            self.setPath(path)
            self.update()

    def paint(self, painter, option, widget=None):
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPath(self.path())

class NodeEdge(QGraphicsPathItem):
    def __init__(self, upper_port, lower_port, parent=None, scene=None):
        super().__init__(parent, scene)
        self.upper_port = upper_port
        self.lower_port = lower_port
        self.source_pos = None
        self.des_pos = None
        self.setZValue(0)
        self.updatePath()
        self.setFlags(QGraphicsItem.ItemIsSelectable)

    def updatePath(self):
        self.source_pos = self.upper_port.getPortPos()
        self.des_pos = self.lower_port.getPortPos()
        
        control_point1 = QPointF((self.source_pos.x() + self.des_pos.x()) / 2, self.source_pos.y())
        control_point2 = QPointF((self.source_pos.x() + self.des_pos.x()) / 2, self.des_pos.y())

        path = QPainterPath(self.source_pos)
        path.cubicTo(control_point1, control_point2, self.des_pos)
        self.setPath(path)
        self.update()

    def paint(self, painter, option, widget=None):
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPath(self.path())
