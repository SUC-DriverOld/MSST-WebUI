from PySide6.QtCore import QRectF, Qt, QPropertyAnimation, QPointF, Property, QObject, Signal
from PySide6.QtWidgets import QGraphicsItem, QApplication, QGraphicsScene, QGraphicsView
from PySide6.QtGui import QBrush, QPen, QColor, QPainter
# import sys
# sys.path.append("/home/tong/projects/python/MSST-WebUI")
from ComfyUI.Editor.common.config import color

class SwitchButton(QGraphicsItem, QObject):
    stateChanged = Signal(bool)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        QGraphicsItem.__init__(self, parent)

        self.width = 30  # 宽度
        self.height = 15  # 高度
        self.margin = 2   # 圆形按钮的边距
        self.is_on = False  # 开关状态
        self._knob_pos = self.margin  # 圆形按钮的 x 轴位置
        self.knob_radius = self.height / 2 - self.margin  # 圆形按钮的半径

        # 动画
        self.animation = QPropertyAnimation(self, b"knobPosition")
        self.animation.setDuration(200)

        # 禁用点击事件标志
        self.is_animation_running = False

        self.animation.finished.connect(self.onAnimationFinished)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget=None):
        # 绘制背景
        bg_color = color if self.is_on else QColor("#282828")
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.boundingRect(), self.height / 2, self.height / 2)

        # 绘制圆形按钮
        knob_color = QColor("#212121")
        painter.setBrush(QBrush(knob_color))
        painter.setPen(QPen(QColor("#808080"), 1))
        painter.drawEllipse(QPointF(self._knob_pos + self.knob_radius, self.height / 2),
                            self.knob_radius, self.knob_radius)

    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if not self.is_animation_running:
            self.toggle()

        super().mousePressEvent(event)    

    def toggle(self):
        """切换状态并启动动画"""
        self.is_on = not self.is_on
        self.is_animation_running = True

        start_pos = self._knob_pos
        end_pos = self.width - self.height + self.margin if self.is_on else self.margin

        self.animation.setStartValue(start_pos)
        self.animation.setEndValue(end_pos)
        self.animation.start()

        self.stateChanged.emit(self.is_on)

    def onAnimationFinished(self):
        """动画完成时允许点击"""
        self.is_animation_running = False

    def getKnobPosition(self):
        return self._knob_pos

    def setKnobPosition(self, pos):
        """更新圆形按钮位置"""
        self._knob_pos = pos
        self.update()

    def isOn(self):
        return self.is_on
    
    def setToggled(self, state: bool):
        """设置开关状态并触发动画"""
        if self.is_on != state:
            self.is_on = state
            self.is_animation_running = True

            start_pos = self._knob_pos
            end_pos = self.width - self.height + self.margin if self.is_on else self.margin

            self.animation.setStartValue(start_pos)
            self.animation.setEndValue(end_pos)
            self.animation.start()

            self.stateChanged.emit(self.is_on)


    knobPosition = Property(float, getKnobPosition, setKnobPosition)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    view.setRenderHint(QPainter.Antialiasing)
    view.setSceneRect(0, 0, 400, 300)

    switch = SwitchButton()
    scene.addItem(switch)
    switch.setPos(50, 50)

    switch.stateChanged.connect(lambda state: print(f"Switch is {'On' if state else 'Off'}"))

    view.show()
    sys.exit(app.exec())
