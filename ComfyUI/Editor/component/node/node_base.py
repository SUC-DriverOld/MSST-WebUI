from PySide6.QtWidgets import QGraphicsItem, QGraphicsDropShadowEffect
from PySide6.QtGui import QFont, QPen, QFontMetrics, QBrush, QColor, QPainterPath
from PySide6.QtCore import Qt, QRectF
import json
import abc

class NodeBase(QGraphicsItem, abc.ABC):
    def __init__(self, parent = None, width = 200, height = 30, node_dict = None):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.node_dict = node_dict if node_dict else {}
        self.input_ports = []
        self.output_ports = []
        self.parameter_ports = []
        self.bool_ports = []
        self.edges = []
        self.add_ports()
        self.init_shadow()
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)

    @abc.abstractmethod
    def add_ports(self):
        pass

    def init_shadow(self):
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(20)
        self.shadow_color = QColor('#aaeeee00')