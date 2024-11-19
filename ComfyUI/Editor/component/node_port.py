from PySide6.QtWidgets import QWidget, QGraphicsItem
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QFont, QFontMetrics
import sys
from PySide6.QtWidgets import QApplication, QHBoxLayout
sys.path.append("D:\projects\python\MSST-WebUI")
# for test
from ComfyUI.Editor.common.config import cfg
color = cfg.get(cfg.themeColor)
font = QFont("Consolas", 12)

class InputPort(QGraphicsItem):
    def __init__(self, parent=None, text="just for test"):
        super().__init__(parent)
        self.width = 100
        self.height = 20
        self.is_connected = False
        self.text = text

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def paint(self, painter, option, widget):
        self.polygon = QPolygonF([
            QPointF(2.5, 2.5),
            QPointF(10, 2.5),
            QPointF(17.5, 10),
            QPointF(10, 17.5),
            QPointF(2.5, 17.5)
        ])

        pen = QPen(color)  # 边框颜色
        pen.setWidth(2)  # 边框宽度
        painter.setPen(pen)

        # 设置填充颜色
        if self.is_connected:
            painter.setBrush(QBrush(color))
        else:
            painter.setBrush(QBrush(Qt.transparent))

        painter.drawPolygon(self.polygon)    

        painter.setFont(font)
        painter.setPen(QPen(Qt.white)) # white
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.text)  # 计算文本宽度
        max_text_width = 75 # 最大文本宽度 100 - 20 - 2.5 * 2
        if text_width > max_text_width:
            truncated_text = font_metrics.elidedText(self.text, Qt.ElideRight, max_text_width)
        else:
            truncated_text = self.text
        
        text_height = font_metrics.height()
        text_rect = QRectF(22.5, (self.height - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

    def setConnectionState(self, connected):
        """设置连接状态，并触发重绘"""
        self.is_connected = connected
        self.update() 

class OutputPort(QGraphicsItem):
    def __init__(self, parent=None, text="just for test"):
        super().__init__(parent)
        self.width = 100
        self.height = 20
        self.is_connected = False
        self.text = text

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def paint(self, painter, option, widget):
        self.polygon = QPolygonF([
            QPointF(82.5, 2.5),
            QPointF(90, 2.5),
            QPointF(97.5, 10),
            QPointF(90, 17.5),
            QPointF(82.5, 17.5)
        ])

        pen = QPen(color)  # 边框颜色
        pen.setWidth(2)  # 边框宽度
        painter.setPen(pen)

        # 设置填充颜色
        if self.is_connected:
            painter.setBrush(QBrush(color))
        else:
            painter.setBrush(QBrush(Qt.transparent))

        painter.drawPolygon(self.polygon)    

        painter.setFont(font)
        painter.setPen(QPen(Qt.white)) # white
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.text)  # 计算文本宽度
        max_text_width = 75 # 最大文本宽度 100 - 20 - 2.5 * 2
        if text_width > max_text_width:
            truncated_text = font_metrics.elidedText(self.text, Qt.ElideRight, max_text_width)
        else:
            truncated_text = self.text

        # Calculate vertical centering and draw the text
        text_height = font_metrics.height()
        text_rect = QRectF(77.5 - min(max_text_width, text_width), (self.height - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

    def setConnectionState(self, connected):
        """设置连接状态，并触发重绘"""
        self.is_connected = connected
        self.update() 


class ParameterPort(QGraphicsItem):
    def __init__(self, parent=None, parameter="just for test", default_value=None, type="int", max_value=None, min_value=None, current_value=None):
        super().__init__(parent)
        self.width = 200
        self.height = 20
        self.parameter = parameter
        self.default_value = default_value
        self.parameter_type = type
        self.max_value = max_value
        self.min_value = min_value
        self.current_value = current_value if current_value is not None else default_value

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def paint(self, painter, option, widget):
        self.text = f"{self.parameter}: {self.current_value}"
        painter.setFont(font)
        painter.setPen(QPen(Qt.white)) # white
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.text)  # 计算文本宽度
        max_text_width = 195 # 最大文本宽度 200 - 20 - 2.5 * 2
        if text_width > max_text_width:
            truncated_text = font_metrics.elidedText(self.text, Qt.ElideMiddle, max_text_width)
        else:
            truncated_text = self.text

        # Calculate vertical centering and draw the text
        text_height = font_metrics.height()
        text_rect = QRectF(2.5, (self.height - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            parent_view = self.scene().views()[0]
            
        return super().mousePressEvent(event)    

        
if __name__ == "__main__":
    from PySide6.QtWidgets import QGraphicsScene, QGraphicsView
    app = QApplication(sys.argv)
    
    # 创建一个场景和视图
    scene = QGraphicsScene()
    view = QGraphicsView(scene)

    # 创建 InputPort 和 OutputPort 实例并添加到场景
    input_port = InputPort(text="Input")
    output_port = OutputPort(text="Output Port")
    input_port.setPos(0, 10)
    output_port.setPos(100, 10)

    scene.addItem(input_port)
    
    scene.addItem(output_port)

    # 设置视图和窗口
    view.setRenderHint(QPainter.Antialiasing)  # 开启抗锯齿效果
    view.setRenderHint(QPainter.SmoothPixmapTransform)
    view.setScene(scene)
    view.setSceneRect(scene.itemsBoundingRect())  # 自适应场景大小
    view.show()
    

    sys.exit(app.exec())        