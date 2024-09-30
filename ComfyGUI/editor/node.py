import sys

sys.path.append('.')

from PySide6.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsDropShadowEffect
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QFont
from node_port import NodePort, InputPort, OutputPort, ParamPort, BoolPort
from config import EditorConfig, NodeConfig


class Node(QGraphicsItem):
    port_icon_size = NodeConfig.port_icon_size
    node_width_min = NodeConfig.node_width_min
    node_height_min = NodeConfig.node_height_min
    node_radius = NodeConfig.node_radius
    port_space = NodeConfig.port_space
    port_padding = NodeConfig.port_padding
    title_height = NodeConfig.title_height
    title_padding = NodeConfig.title_padding

    def __init__(self, title = '', input_ports = None, param_ports = None, output_ports = None, bool_ports = None, scene = None, parent = None, upstream_node = None, downstream_nodes = None, index = 1):
        super().__init__(parent)

        self.index = index
        # self._model_type = None
        # self._model = None
        # self._model_config = None
        self._title = title
        self._scene = scene
        self.input_ports = input_ports or []
        self.param_ports = param_ports or []
        self.output_ports = output_ports or []
        self.bool_ports = bool_ports or []
        self.upstream_node = upstream_node
        self.downstream_nodes = downstream_nodes or []
        self.upstream_edges = []
        self.downstream_edges = []
        self._node_width = self.node_width_min
        self._node_height = self.node_height_min

        self._shadow = QGraphicsDropShadowEffect()
        self._shadow.setOffset(0, 0)
        self._shadow.setBlurRadius(20)
        self._shadow_color = QColor('#aaeeee00')

        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)

    def init_node_color(self):
        self._pen_selected = QPen(QColor('#ddffee00'))
        self._brush_background = QBrush(QColor('#dd151515'))
        self._title_bak_color = '#39c5bb'
        title_color = QColor(self._title_bak_color)
        self._pen_default = QPen(title_color)
        title_color.setAlpha(200)
        self._brush_title_back = QBrush(title_color)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._node_width, self._node_height)

    def paint(self, painter: QPainter, option, widget):
        self._shadow.setColor('#00000000')
        self.setGraphicsEffect(self._shadow)

        if self.isSelected():
            self._shadow.setColor(self._shadow_color)
            self.setGraphicsEffect(self._shadow)

        node_outline = QPainterPath()
        node_outline.addRoundedRect(0, 0, self._node_width, self._node_height, self.node_radius, self.node_radius)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._brush_background)
        painter.drawPath(node_outline.simplified())

        title_outline = QPainterPath()
        title_outline.setFillRule(Qt.WindingFill)
        title_outline.addRoundedRect(0, 0, self._node_width, self.title_height, self.node_radius, self.node_radius)
        title_outline.addRect(0, self.title_height - self.node_radius, self.node_radius, self.node_radius)
        title_outline.addRect(self._node_width - self.node_radius, self.title_height - self.node_radius, self.node_radius, self.node_radius)

        painter.setPen(Qt.NoPen)
        painter.setBrush(self._brush_title_back)
        painter.drawPath(title_outline.simplified())

        painter.setPen(self._pen_selected if self.isSelected() else self._pen_default)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(node_outline)

    def set_scene(self, scene):
        self._scene = scene

    def update_ports(self):
        self.init_ports()
        for port_list in [self.input_ports, self.param_ports, self.bool_ports, self.output_ports]:
            for i, port in enumerate(port_list):
                self.add_port(port, index = i)

    def add_port(self, port: NodePort, index = 0):
        self._node_width = max(self._node_width, port._port_width + self.port_padding * 2)
        self._node_height = self.title_height + (max(len(self.input_ports), len(self.output_ports)) + len(self.param_ports) + len(self.bool_ports)) * (self.port_padding + port._port_icon_size) + self.port_padding
        port.add_to_parent_node(self, self._scene)

        y_offset = self.title_height + index * (self.port_padding + port._port_icon_size) + self.port_padding
        if port.port_type == NodePort.PORT_TYPE_INPUT:
            port.setPos(self.port_padding, y_offset)
        elif port.port_type == NodePort.PORT_TYPE_OUTPUT:
            port.setPos(self._node_width - port._port_width - self.port_padding, y_offset)
        elif port.port_type == NodePort.PORT_TYPE_PARAM:
            port.setPos(self.port_padding, y_offset + max(len(self.input_ports), len(self.output_ports)) * (self.port_padding + port._port_icon_size))
        elif port.port_type == NodePort.PORT_TYPE_BOOL:
            port.setPos(self.port_padding, y_offset + (len(self.param_ports) + max(len(self.input_ports), len(self.output_ports))) * (self.port_padding + ParamPort()._port_icon_size))

    def remove_edge(self):
        for port in self.input_ports + self.output_ports + self.param_ports + self.bool_ports:
            print('Removing edge from port:', port)
            port.remove_edge()
            port.update()
