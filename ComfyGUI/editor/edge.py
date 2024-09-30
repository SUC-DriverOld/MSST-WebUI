# coding:utf-8
'''
节点的连接边
'''
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QGraphicsDropShadowEffect
from PySide6.QtGui import QPen, QColor, QPainterPath, QPainter
from PySide6.QtCore import Qt, QPointF
from node_port import NodePort


class BaseEdge(QGraphicsPathItem):

    def __init__(self, source_pos, des_pos, scene = None, edge_color = '#ffffff', parent = None):
        super().__init__(parent)
        self._source_pos = source_pos
        self._des_pos = des_pos
        self._scene = scene
        self._edge_color = edge_color
        self._pen_default = QPen(QColor(self._edge_color))
        self._pen_default.setWidthF(2)
        self.setZValue(-1)  # 设置层级为最底层

    def paint(self, painter: QPainter, option, widget):
        self.update_edge_path()
        painter.setPen(self._pen_default)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    def update_edge_path(self):
        source_pos = self._source_pos
        des_pos = self._des_pos
        self._path = QPainterPath(source_pos)

        control_offset_x = abs(des_pos.x() - source_pos.x()) * 0.6
        control_offset_y = abs(des_pos.y() - source_pos.y()) * 0.1

        if source_pos.x() < des_pos.x():
            control1 = QPointF(source_pos.x() + control_offset_x, source_pos.y() + control_offset_y)
            control2 = QPointF(des_pos.x() - control_offset_x, des_pos.y() - control_offset_y)
        else:
            control1 = QPointF(source_pos.x() - control_offset_x, source_pos.y() + control_offset_y)
            control2 = QPointF(des_pos.x() + control_offset_x, des_pos.y() - control_offset_y)
        self._path.cubicTo(control1, control2, des_pos)

        pen = QPen(QColor(self._edge_color), 2)
        self.setPen(pen)
        self.setPath(self._path)


class NodeEdge(BaseEdge):

    def __init__(self, source_port, des_port, scene = None, edge_color = '#ffffff', parent = None):
        source_pos = source_port.get_port_pos()
        des_pos = des_port.get_port_pos()
        super().__init__(source_pos, des_pos, scene, edge_color, parent)

        self._source_port = source_port
        self._des_port = des_port

        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self._shadow = QGraphicsDropShadowEffect()
        self._shadow.setOffset(0, 0)
        self._shadow.setBlurRadius(20)
        self._shadow_color = Qt.yellow

        self.add_to_scene()
        self._source_port.update()
        self._des_port.update()

    def add_to_scene(self):
        self._scene.addItem(self)
        self._source_port.add_edge(self, self._des_port)
        self._des_port.add_edge(self, self._source_port)

    def paint(self, painter: QPainter, option, widget):
        super().paint(painter, option, widget)

        if self.isSelected():
            self._shadow.setColor(self._shadow_color)
            self.setGraphicsEffect(self._shadow)
        else:
            self.setGraphicsEffect(None)

    def update_edge_path(self):
        self._source_pos = self._source_port.get_port_pos()
        self._des_pos = self._des_port.get_port_pos()
        super().update_edge_path()


class DraggingEdge(BaseEdge):

    def __init__(self, source_pos, des_pos, drag_from_source = True, scene = None, edge_color = '#ffffff', parent = None):
        super().__init__(source_pos, des_pos, scene, edge_color, parent)
        self._drag_from_source = drag_from_source

    def update_position(self, pos):
        if self._drag_from_source:
            self._source_pos = pos
        else:
            self._des_pos = pos
        self.prepareGeometryChange()
        self.update_edge_path()
        self.update()

    def set_first_port(self, port):
        if self._drag_from_source:
            self._source_port = port
        else:
            self._des_port = port

    def set_second_port(self, port):
        if not self._drag_from_source:
            self._source_port = port
        else:
            self._des_port = port

    def create_node_edge(self):
        if self.check_ports(self._source_port, self._des_port):
            return NodeEdge(self._source_port, self._des_port, self._scene)
        return None

    def check_ports(self, source_port, des_port):

        # 仅允许连接输入和输出端口
        if source_port.port_type == NodePort.PORT_TYPE_INPUT and des_port.port_type == NodePort.PORT_TYPE_OUTPUT:
            return True
        if source_port.port_type == NodePort.PORT_TYPE_OUTPUT and des_port.port_type == NodePort.PORT_TYPE_INPUT:
            return True

        return False
