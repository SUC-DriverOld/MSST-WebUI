from PySide6.QtWidgets import QGraphicsItem
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QFont, QFontMetrics
# import sys
# from PySide6.QtWidgets import QApplication, QHBoxLayout
# sys.path.append("D:\projects\python\MSST-WebUI")
# for test
from ComfyUI.Editor.common.config import cfg
from ComfyUI.Editor.component.graphic_switch_button import SwitchButton
from ComfyUI.Editor.component.parameter_message_box import ParameterMessageBox
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
    def __init__(self, parent=None, parameter="just for test", default_value=None, type=int, max_value=None, min_value=None, current_value=None):
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

        text_height = font_metrics.height()
        text_rect = QRectF(2.5, (self.height - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

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
                    else:
                        return
                    self.current_value = new_value
                    self.update()
                except ValueError:
                    pass
            
        return super().mousePressEvent(event)  
    
    def setValue(self, value):
        self.current_value = value
        self.update()
    
    
class BoolPort(QGraphicsItem):
    def __init__(self, parent=None, parameter="just for test", default_value=False, current_value=None):
        super().__init__(parent)
        self.width = 200
        self.height = 20
        self.parameter = parameter
        self.default_value = default_value
        self.current_value = current_value if current_value is not None else default_value
        self.switch_button = SwitchButton()
        self.switch_button.setPos(2.5, 2.5)
        self.switch_button.setParentItem(self)
        self.switch_button.stateChanged.connect(self.onStateChanged)
        
    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)
    
    def paint(self, painter, option, widget):
        
        self.text = f"{self.parameter}"
        painter.setFont(font)
        painter.setPen(QPen(Qt.white)) # white
        font_metrics = QFontMetrics(font)
        text_width = font_metrics.horizontalAdvance(self.text)  # 计算文本宽度
        max_text_width = 200 - 37.5 # 最大文本宽度
        if text_width > max_text_width:
            truncated_text = font_metrics.elidedText(self.text, Qt.ElideMiddle, max_text_width)
        else:
            truncated_text = self.text

        # Calculate vertical centering and draw the text
        text_height = font_metrics.height()
        text_rect = QRectF(35, (self.height - text_height) / 2, max_text_width, text_height)
        painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

    def onStateChanged(self, state):
        self.current_value = state
        print(f"{self.parameter} state changed to {self.current_value}")
        # self.update()    


class FormatSelector(QGraphicsItem):
    def __init__(self, options=None, parent=None):
        super().__init__(parent)
        self.options = options if options else ["wav", "mp3", "flac"]
        self.currentIndex = 0  # 默认选中第一个选项
        self.itemRadius = 8    # 单个选项圆圈的半径
        self.spacing = 66      # 每个选项之间的水平间距
        self.width = 200
        self.height = 20
        self.font = font
        self.color = color
        self.initItems()

    def initItems(self):
        """初始化选择器"""
        self.updateSelection()

    def updateSelection(self):
        """更新当前选中的样式"""
        self.update()

    def boundingRect(self) -> QRectF:
        """定义组件的边界"""
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget=None):
        """绘制选择器"""
        painter.setFont(self.font)
        font_metrics = QFontMetrics(self.font)

        for i, option in enumerate(self.options):
            circle_x = 10 + i * self.spacing
            circle_y = self.height / 2 - self.itemRadius

            painter.setPen(QPen(QColor(100, 100, 100), 1))
            if i == self.currentIndex:
                painter.setBrush(QBrush(self.color))
            else:
                painter.setBrush(QBrush(Qt.white))
            painter.drawEllipse(circle_x, circle_y, 2 * self.itemRadius, 2 * self.itemRadius)

            text_width = font_metrics.horizontalAdvance(option)
            text_height = font_metrics.height()

            max_text_width = self.spacing - 2 * self.itemRadius - 10
            if text_width > max_text_width:
                truncated_text = font_metrics.elidedText(option, Qt.ElideRight, max_text_width)
            else:
                truncated_text = option

            text_x = circle_x + 2 * self.itemRadius + 5
            text_y = (self.height - text_height) / 2

            painter.setPen(QPen(self.color))
            text_rect = QRectF(text_x, text_y, max_text_width, text_height)
            painter.drawText(text_rect, Qt.AlignVCenter, truncated_text)

    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        pos = event.pos()
        for i in range(len(self.options)):
            circle_x = 10 + i * self.spacing
            circle_y = self.height / 2 - self.itemRadius
            circle_rect = QRectF(circle_x, circle_y, 2 * self.itemRadius, 2 * self.itemRadius)

            if circle_rect.contains(pos):
                self.currentIndex = i
                self.updateSelection()
                break
        super().mousePressEvent(event)

    def select(self, index):
        """选择某个选项"""
        if 0 <= index < len(self.options):
            self.currentIndex = index
            self.updateSelection()

    def getSelectedValue(self):
        """返回当前选择的值"""
        return self.options[self.currentIndex]

        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    from PySide6.QtWidgets import QGraphicsScene, QGraphicsView
    # 创建场景和视图
    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    view.setFixedSize(220, 60)  # 固定视图大小，稍微多出一些边距

    # 添加 FormatSelector
    selector = FormatSelector(["wav", "mp3", "flac"])
    scene.addItem(selector)
    selector.setPos(10, 20)  # 定位组件到场景中

    # 绑定鼠标点击后打印当前选项的功能
    # def mousePressEvent(event):
    #     # 调用原始事件处理逻辑
    #     QGraphicsScene.mousePressEvent(scene, event)
    #     # 打印当前选择的值
    #     print("当前选择的格式:", selector.getSelectedValue())

    # # 重写鼠标事件
    # scene.mousePressEvent = mousePressEvent

    view.show()
    sys.exit(app.exec())
