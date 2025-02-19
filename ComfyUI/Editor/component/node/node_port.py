from PySide6.QtWidgets import QGraphicsItem
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QBrush, QPen, QPolygonF, QFontMetrics
from ComfyUI.Editor.common.config import color, font
from ComfyUI.Editor.component.graphic_switch_button import SwitchButton
from ComfyUI.Editor.component.parameter_message_box import ParameterMessageBox

class BasePort(QGraphicsItem):
    def __init__(self, parent=None, width=100, height=20):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.parent_node = None
        self.index = -1

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def setParentNode(self, node, index=-1):
        self.parent_node = node
        self.index = index

    def _draw_text(self, painter, text, max_width, x_offset):
        painter.setFont(font)
        painter.setPen(QPen(Qt.white))
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(text)
        if text_width > max_width:
            truncated_text = font_metrics.elidedText(text, Qt.ElideRight, max_width)
        else:
            truncated_text = text
        text_height = font_metrics.height()
        text_rect = QRectF(x_offset, (self.height - text_height) / 2, max_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

class ConnectionPort(BasePort):
    def __init__(self, parent=None, text="just for test", width=100, height=20, is_input=True):
        super().__init__(parent, width, height)
        self.is_connected = False
        self.text = text
        self.connected_edges = []
        self.is_input = is_input

    def paint(self, painter, option, widget):
        if self.is_input:
            polygon_points = [
                QPointF(2.5, 2.5),
                QPointF(10, 2.5),
                QPointF(17.5, 10),
                QPointF(10, 17.5),
                QPointF(2.5, 17.5)
            ]
            x_offset = 22.5
            max_text_width = 75
        else:
            polygon_points = [
                QPointF(82.5, 2.5),
                QPointF(90, 2.5),
                QPointF(97.5, 10),
                QPointF(90, 17.5),
                QPointF(82.5, 17.5)
            ]
            x_offset = 77.5 - 75
            max_text_width = 75

        self.polygon = QPolygonF(polygon_points)
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(color if self.is_connected else Qt.transparent))
        painter.drawPolygon(self.polygon)
        self._draw_text(painter, self.text, max_text_width, x_offset)

    def updateConnectionState(self):
        self.is_connected = len(self.connected_edges) > 0
        self.update()

    def getPortPos(self):
        if self.is_input:
            return self.scenePos() + QPointF(2.5, self.height / 2)
        else:
            return self.scenePos() + QPointF(97.5, self.height / 2)

    def setPosition(self, x: float, y: float):
        self.setPos(QPointF(x, y))
        for edge in self.connected_edges:
            edge.updatePath()

    def addConnectedEdge(self, edge):
        self.connected_edges.append(edge)

class InputPort(ConnectionPort):
    def __init__(self, parent=None, text="just for test", width=100, height=20):
        super().__init__(parent, text, width, height, is_input=True)

class OutputPort(ConnectionPort):
    def __init__(self, parent=None, text="just for test", width=100, height=20):
        super().__init__(parent, text, width, height, is_input=False)

class ValuePort(BasePort):
    def __init__(self, parent=None, width=200, height=20, parameter="just for test", default_value=None):
        super().__init__(parent, width, height)
        self.parameter = parameter
        self.default_value = default_value
        self.current_value = default_value

    def _draw_value_text(self, painter, text):
        self._draw_text(painter, text, 195, 2.5)

class ParameterPort(ValuePort):
    def __init__(self, parent=None, parameter="just for test", default_value=None, type=int, max_value=None, min_value=None, current_value=None):
        super().__init__(parent, parameter=parameter, default_value=default_value)
        self.parameter_type = type
        self.max_value = max_value
        self.min_value = min_value
        self.current_value = current_value if current_value is not None else default_value

    def paint(self, painter, option, widget):
        text = f"{self.parameter}: {self.current_value}"
        self._draw_value_text(painter, text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            parent_view = self.scene().views()[0]
            w = ParameterMessageBox(
                parent=parent_view,
                parameter=self.parameter,
                default_value=self.default_value,
                current_value=self.current_value,
                type=self.parameter_type,
                max_value=self.max_value,
                min_value=self.min_value
            )

            if w.exec():
                try:
                    if self.parameter_type == int:
                        new_value = int(w.LineEdit.text())
                    elif self.parameter_type == float:
                        new_value = float(w.LineEdit.text())
                    elif self.parameter_type == str:
                        new_value = str(w.LineEdit.text())
                    else:
                        return
                    self.setValue(new_value)
                except ValueError:
                    pass

        super().mousePressEvent(event)

    def setValue(self, value):
        self.current_value = value
        self.parent_node.node_dict["parameter"][self.index]["current_value"] = value
        self.update()

class BoolPort(ValuePort):
    def __init__(self, parent=None, parameter="just for test", default_value=False, current_value=None):
        super().__init__(parent, parameter=parameter, default_value=default_value)
        self.current_value = current_value if current_value is not None else default_value
        self.switch_button = SwitchButton()
        self.switch_button.setPos(2.5, 2.5)
        self.switch_button.setParentItem(self)
        self.switch_button.stateChanged.connect(self.setValue)


    def paint(self, painter, option, widget):
        text = f"{self.parameter}"
        self._draw_text(painter, text, 162.5, 35)

    def setValue(self, value):
        self.current_value = value
        self.parent_node.node_dict["bool"][self.index]["current_value"] = value
        self.switch_button.setToggled(value)
        self.update()