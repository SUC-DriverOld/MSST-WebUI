from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, QRect
from PySide6.QtGui import QPainter, QBrush, QPen
# import sys
# from PySide6.QtWidgets import QApplication, QHBoxLayout
# sys.path.append("/home/tong/projects/python/MSST-WebUI")
# for test
from qfluentwidgets import CaptionLabel
from ComfyUI.Editor.common.config import cfg
color = cfg.get(cfg.themeColor)

class InputPort(QWidget):
    def __init__(self, parent=None, connected=False, text=""):
        super().__init__(parent)
        self.setFixedSize(100, 20)
        self.connected = connected
        self.label = CaptionLabel(text, self)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.move(25, 0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        circle_center = QPointF(10, self.height() / 2)
        circle_radius = 8

        circle_rect = QRect(circle_center.x() - circle_radius, circle_center.y() - circle_radius, circle_radius * 2, circle_radius * 2)
        
        pen = QPen(color, 2)
        painter.setPen(pen)
        if self.connected:
            painter.setBrush(QBrush(color))
        else:
            painter.setBrush(Qt.transparent)
            painter.setPen(QPen(color, 2, Qt.DashLine))  # 未连接时，使用虚线边框

        painter.drawEllipse(circle_rect)

    def updateCircle(self):
        circle_center = QPointF(15, self.height() / 2)
        circle_radius = 8
        circle_rect = QRect(circle_center.x() - circle_radius, circle_center.y() - circle_radius, circle_radius * 2, circle_radius * 2)
        # 只更新圆圈区域
        self.update(circle_rect)

    @property
    def center(self):
        """返回圆圈的中心点"""
        return QPointF(15, self.height() / 2)

    def setConnected(self, connected):
        """设置连接状态并触发圆圈更新"""
        if self.connected != connected:
            self.connected = connected
            self.updateCircle()


class OutputPort(QWidget):
    def __init__(self, parent=None, connected=False, text=""):
        super().__init__(parent)
        self.setFixedSize(100, 20)
        self.connected = connected
        self.label = CaptionLabel(text, self)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.adjustSize()
        # print(self.label.width())
        self.label.move(75 - self.label.width(), 0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        circle_center = QPointF(self.width() - 10, self.height() / 2)
        circle_radius = 8

        circle_rect = QRect(circle_center.x() - circle_radius, circle_center.y() - circle_radius, circle_radius * 2, circle_radius * 2)

        pen = QPen(color, 2)
        painter.setPen(pen)

        if self.connected:
            painter.setBrush(QBrush(color))
        else:
            painter.setBrush(Qt.transparent)
            painter.setPen(QPen(color, 2, Qt.DashLine))

        painter.drawEllipse(circle_rect)

    def updateCircle(self):
        circle_center = QPointF(self.width() - 15, self.height() / 2)
        circle_radius = 8
        circle_rect = QRect(circle_center.x() - circle_radius, circle_center.y() - circle_radius, circle_radius * 2, circle_radius * 2)
        self.update(circle_rect)

    @property
    def center(self):
        """返回圆圈的中心点"""
        return QPointF(self.width() - 15, self.height() / 2)

    def setConnected(self, connected):
        """设置连接状态并触发圆圈更新"""
        if self.connected != connected:
            self.connected = connected
            self.updateCircle()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window_layout = QHBoxLayout(window)
    input_port = InputPort(text="Input")
    output_port = OutputPort(text="Output")
    window_layout.addWidget(input_port)
    window_layout.addWidget(output_port)
    window.show()
    sys.exit(app.exec())