from PySide6.QtWidgets import QGraphicsItem, QGraphicsDropShadowEffect
from PySide6.QtGui import QFont, QPen, QFontMetrics, QBrush, QColor, QPainterPath
from PySide6.QtCore import Qt, QRectF
import json
import abc
from ComfyUI.Editor.common.config import font, color

class NodeBase(QGraphicsItem):
    def __init__(self, parent = None, width = 200, height = 30, node_dict = None, title = "Node", subtitle = "Node"):
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
        self.title = title
        self.subtitle = subtitle
        self.init_title()
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)

    def to_dict(self):
        pass

    def add_ports(self):
        pass

    def init_shadow(self):
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(20)
        self.shadow_color = QColor('#aaeeee00')

    def init_title(self):
        self.title_font = font
        self.title_font.setBold(True)
        self.title_font.setPointSize(10)
        
        self.subtitle_font = font
        self.subtitle_font.setPointSize(8)
        
        self.title_text_color = QColor(255, 255, 255)
        
        self.title_background_color = color
        
        self.corner_radius = 8
        
        self.title_height = 30
        self.title_rect = QRectF(0, 0, self.width, self.title_height)
        
        self.title_margin = 10
        self.title_pos_x = self.title_margin
        
        title_metrics = QFontMetrics(self.title_font)
        subtitle_metrics = QFontMetrics(self.subtitle_font)
        
        if hasattr(self, 'subtitle') and self.subtitle:
            self.title_pos_y = self.title_height * 0.35 + title_metrics.height() / 4
            self.subtitle_pos_y = self.title_height * 0.7 + subtitle_metrics.height() / 4
        else:
            self.title_pos_y = self.title_height / 2 + title_metrics.height() / 4
            
        self.available_width = self.width - 2 * self.title_margin
        
        self.elided_title = self._get_elided_text(self.title, self.title_font, self.available_width)
        if hasattr(self, 'subtitle') and self.subtitle:
            self.elided_subtitle = self._get_elided_text(self.subtitle, self.subtitle_font, self.available_width)

    def _get_elided_text(self, text, font, width):
        metrics = QFontMetrics(font)
        
        if metrics.horizontalAdvance(text) <= width:
            return text
        
        ellipsis = "..."
        ellipsis_width = metrics.horizontalAdvance(ellipsis)
        
        available_width = width - ellipsis_width
        
        if available_width <= 0:
            return ""
        
        half_width = available_width // 2
        
        left_part = ""
        for char in text:
            if metrics.horizontalAdvance(left_part + char) > half_width:
                break
            left_part += char
        
        right_part = ""
        for char in reversed(text):
            if metrics.horizontalAdvance(right_part + char) > half_width:
                break
            right_part = char + right_part
        
        return left_part + ellipsis + right_part
            
    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        painter.setRenderHint(painter.Antialiasing)
        
        background_path = QPainterPath()
        background_path.addRoundedRect(QRectF(0, 0, self.width, self.height), 
                                    self.corner_radius, self.corner_radius)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(40, 40, 50)))
        painter.drawPath(background_path)
        
        title_path = QPainterPath()
        title_path.addRoundedRect(self.title_rect, 
                                self.corner_radius, self.corner_radius)
        
        if self.height > self.title_height:
            clip_rect = QRectF(0, self.title_height/2, self.width, self.title_height/2)
            clip_path = QPainterPath()
            clip_path.addRect(clip_rect)
            title_path = title_path.subtracted(clip_path)
        
        painter.setBrush(QBrush(self.title_background_color))
        painter.drawPath(title_path)
        
        painter.setPen(self.title_text_color)
        painter.setFont(self.title_font)
        painter.drawText(self.title_pos_x, self.title_pos_y, self.title)
        
        if self.isSelected():
            select_pen = QPen(QColor(255, 255, 0))
            select_pen.setWidth(2)
            painter.setPen(select_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(0, 0, self.width, self.height), 
                                self.corner_radius, self.corner_radius)
            
    def addItem(self, item: QGraphicsItem):
        item.setParentItem(self)        