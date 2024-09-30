import sys

sys.path.append('.')

from PySide6.QtWidgets import QGraphicsItem, QLineEdit, QGraphicsProxyWidget, QCheckBox
from PySide6.QtGui import QColor, QPen, QBrush, QPainter, QFont, QFontMetrics
from PySide6.QtCore import Qt, QRectF, QPointF
from config import EditorConfig, NodeConfig


class NodePort(QGraphicsItem):
    PORT_TYPE_INPUT = 1001
    PORT_TYPE_OUTPUT = 1002
    PORT_TYPE_PARAM = 1003
    PORT_TYPE_BOOL = 1004

    def __init__(self, port_label = '', port_color = '#ffffff', port_type = PORT_TYPE_INPUT, parent = None,
                 edges = None):
        super().__init__(parent)
        self.port_pos = None
        self.parent_node = None
        self._scene = None
        self._port_label = port_label  # 端口标签
        self._port_color = port_color  # 端口颜色
        self.port_type = port_type  # 端口类型：输入、输出、参数、布尔类型

        # 初始化画笔和画刷
        self._pen_default = QPen(QColor(self._port_color))
        self._pen_default.setWidthF(1.5)
        self._brush_default = QBrush(QColor(self._port_color))

        # 字体和大小
        self._port_font = QFont(EditorConfig.editor_node_pin_label_font, EditorConfig.editor_node_pin_label_size)
        self._font_metrics = QFontMetrics(self._port_font)
        self._port_icon_size = NodeConfig.port_icon_size  # 端口图标大小
        self._port_label_size = self._font_metrics.horizontalAdvance(self._port_label)  # 端口标签宽度
        self._port_width = self._port_icon_size + self._port_label_size  # 端口宽度

        self.port_value = None  # 端口值
        self.has_value_set = False  # 是否设置了值

        self.edges = [] if edges is None else edges
        self.connected_ports: list[NodePort] = [] if edges is None else edges

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._port_width, self._port_icon_size)

    def add_to_parent_node(self, parent_node, scene):
        self.setParentItem(parent_node)
        self.parent_node = parent_node
        self._scene = scene

        if self.port_type == NodePort.PORT_TYPE_INPUT:
            self.setPos(0, 0)
        elif self.port_type == NodePort.PORT_TYPE_OUTPUT:
            self.setPos(parent_node.boundingRect().width() - self._port_width, 0)

    def get_port_pos(self):
        self.port_pos = self.scenePos()
        if self.port_type == NodePort.PORT_TYPE_INPUT:
            return QPointF(self.port_pos.x() + 0.5 * self._port_icon_size,
                           self.port_pos.y() + 0.5 * self._port_icon_size)
        elif self.port_type == NodePort.PORT_TYPE_OUTPUT:
            return QPointF(self.port_pos.x() + 0.5 * self._port_icon_size + self._port_label_size + 5,
                           self.port_pos.y() + 0.5 * self._port_icon_size)

    def add_edge(self, edge, connected_port):
        if self.port_type == NodePort.PORT_TYPE_INPUT:
            self.edges.append(edge)
            self.connected_ports.append(connected_port)
            self.parent_node.upstream_node = connected_port.parent_node
            self.parent_node.upstream_edges.append(edge)
        else:
            self.edges.append(edge)
            self.connected_ports.append(connected_port)
            connected_port.parent_node.upstream_node = self.parent_node
            connected_port.parent_node.upstream_edges.append(edge)

    def remove_edge(self):
        for edge in self.edges:
            print('Removing edge:', edge)
            self.parent_node._scene.removeItem(edge)
            edge._des_port.edges.remove(edge)
            edge._source_port.edges.remove(edge)
            edge._des_port.update()
            edge._source_port.update()
            self.update()

    def is_connected(self):
        return len(self.edges) > 0

    def set_port_value(self, value):
        self.port_value = value
        self.has_value_set = True


class InputPort(NodePort):

    def __init__(self, port_label = ''):
        super().__init__(port_label = port_label, port_type = NodePort.PORT_TYPE_INPUT)

    def paint(self, painter: QPainter, option, widget) -> None:
        square = QRectF(0, 0, self._port_icon_size, self._port_icon_size)
        painter.setPen(self._pen_default)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(square)

        if self.is_connected():
            small_square = QRectF(
                self._port_icon_size / 4,
                self._port_icon_size / 4,
                self._port_icon_size / 2,
                self._port_icon_size / 2
            )
            painter.setBrush(self._brush_default)
            painter.drawRect(small_square)

        painter.setFont(self._port_font)
        painter.drawText(QRectF(self._port_icon_size + 5, 0, self._port_label_size, self._port_icon_size),
                         Qt.AlignLeft | Qt.AlignVCenter, self._port_label)


class OutputPort(NodePort):

    def __init__(self, port_label = ''):
        super().__init__(port_label = port_label, port_type = NodePort.PORT_TYPE_OUTPUT)

    def paint(self, painter: QPainter, option, widget) -> None:
        painter.setPen(self._pen_default)
        painter.setFont(self._port_font)
        painter.drawText(QRectF(0, 0, self._port_label_size, self._port_icon_size),
                         Qt.AlignRight | Qt.AlignVCenter, self._port_label)

        square = QRectF(self._port_label_size + 5, 0, self._port_icon_size, self._port_icon_size)
        painter.setPen(self._pen_default)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(square)

        if self.is_connected():
            small_square = QRectF(
                self._port_label_size + self._port_icon_size / 4 + 5,
                self._port_icon_size / 4,
                self._port_icon_size / 2,
                self._port_icon_size / 2
            )
            painter.setBrush(self._brush_default)
            painter.drawRect(small_square)


class ParamPort(NodePort):

    def __init__(self, port_label = '', port_color = '#ffffff', default_value = '', parent = None):
        super().__init__(port_label = port_label, port_color = port_color, port_type = NodePort.PORT_TYPE_PARAM,
                         parent = parent)

        self._default_value = default_value
        self._line_edit = QLineEdit()
        self._line_edit.setText(str(self._default_value))

        self._proxy_widget = QGraphicsProxyWidget(self)
        self._proxy_widget.setWidget(self._line_edit)

        self._port_label_size = self._font_metrics.horizontalAdvance(self._port_label)
        self._port_textbox_width = 100
        self._port_width = self._port_label_size + self._port_textbox_width

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._port_width, self._port_icon_size)

    def paint(self, painter: QPainter, option, widget) -> None:
        painter.setPen(self._pen_default)
        painter.setFont(self._port_font)
        painter.drawText(QRectF(0, 0, self._port_label_size, self._port_icon_size),
                         Qt.AlignLeft | Qt.AlignVCenter, self._port_label)

        self._proxy_widget.setGeometry(
            QRectF(self._port_label_size + 5, 0, self._port_textbox_width, self._port_icon_size))

    def get_value(self):
        return self._line_edit.text()


class BoolPort(NodePort):

    def __init__(self, port_label = '', port_color = '#ffffff', default_value = False, parent = None):
        super().__init__(port_label = port_label, port_color = port_color, port_type = NodePort.PORT_TYPE_BOOL,
                         parent = parent)

        self._default_value = default_value

        # 创建复选框并设置默认值
        self._checkbox = QCheckBox()
        self._checkbox.setChecked(self._default_value)

        # 使用 QGraphicsProxyWidget 添加到图形项中
        self._proxy_widget = QGraphicsProxyWidget(self)
        self._proxy_widget.setWidget(self._checkbox)

        self._port_label_size = self._font_metrics.horizontalAdvance(self._port_label)
        self._port_checkbox_width = self._checkbox.sizeHint().width()  # 根据复选框大小调整

        # 重新计算端口宽度
        self._port_width = self._port_label_size + self._port_checkbox_width + 5

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._port_width, self._port_icon_size)

    def paint(self, painter: QPainter, option, widget) -> None:
        painter.setPen(self._pen_default)
        painter.setFont(self._port_font)

        # 绘制端口标签
        painter.drawText(QRectF(0, 0, self._port_label_size, self._port_icon_size),
                         Qt.AlignLeft | Qt.AlignVCenter, self._port_label)

        # 设置复选框位置
        self._proxy_widget.setGeometry(
            QRectF(self._port_label_size + 5, 0, self._port_checkbox_width, self._port_icon_size))

    def get_value(self):
        return self._checkbox.isChecked()
