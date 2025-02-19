from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView, QApplication, QGraphicsPixmapItem
from PySide6.QtCore import Qt, QRectF, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QFont, QPen, QIcon, QPixmap, QFontMetrics
from ComfyUI.Editor.common.config import font
from qfluentwidgets import FluentIcon

class ExpandableItem(QGraphicsItem):
    def __init__(self, text, value_ports, collapsed_height=20, width=200, parent=None):
        super().__init__(parent)
        self.width = width
        self.collapsed_height = collapsed_height
        self.height = self.collapsed_height
        self.is_expanded = False
        self.text = text
        self.value_ports = value_ports

        self.icon = FluentIcon.SCROLL
        self.icon_size = 20
        self.icon_item = QGraphicsPixmapItem(self.icon.pixmap(self.icon_size, QIcon.Normal, QIcon.Off), self) # 初始为折叠状态


        self.animation = QPropertyAnimation(self, b"height")
        self.animation.setDuration(250)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.finished.connect(self.on_animation_finished)

        self._update_value_ports_visibility()

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self._update_layout()

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        pass

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_expand()
        super().mousePressEvent(event)

    def toggle_expand(self):
        self.is_expanded = not self.is_expanded
        total_ports_height = sum(port.height for port in self.value_ports)
        self.animation.setStartValue(self.height)
        self.animation.setEndValue(self.collapsed_height + total_ports_height if self.is_expanded else self.collapsed_height)
        self.animation.start()

        self.icon_item.setPixmap(self.icon.pixmap(self.icon_size, QIcon.Normal, QIcon.On if self.is_expanded else QIcon.Off))
        self._update_layout()
        self.update()


    def _update_value_ports_visibility(self):
        y_offset = self.collapsed_height
        for port in self.value_ports:
            port.setParentItem(self)
            port.setPos(0, y_offset)
            port.setVisible(self.is_expanded)
            y_offset += port.height

    def on_animation_finished(self):
        self._update_value_ports_visibility()
        self.update()
        
    def _update_layout(self):
        # 计算文本宽度
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.text)

        # 计算图标和文本的总宽度
        total_width = self.icon_size + 5 + text_width  # 5 是图标和文本之间的间距

        # 计算起始 x 坐标，使图标和文本整体居中
        start_x = (self.width - total_width) / 2

        # 设置图标位置
        icon_y = (self.collapsed_height - self.icon_size) / 2
        self.icon_item.setPos(start_x, icon_y)
        
        # 设置文本item
        text_height = font_metrics.height()
        text_y = (self.collapsed_height - text_height) / 2
        if not hasattr(self, 'text_item'):
            self.text_item = QGraphicsPixmapItem(self)
        
        text_pixmap = QPixmap(text_width,text_height)
        text_pixmap.fill(Qt.transparent)
        text_painter = QPainter(text_pixmap)
        text_painter.setFont(QFont("Arial", 10))
        text_painter.setPen(QPen(Qt.white))
        text_painter.drawText(QRectF(0, 0, text_width, text_height), Qt.AlignVCenter, self.text)
        text_painter.end()
        
        self.text_item.setPixmap(text_pixmap)
        self.text_item.setPos(start_x + self.icon_size + 5, text_y)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        self.prepareGeometryChange()
        self.update()