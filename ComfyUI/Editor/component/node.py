from PySide6.QtWidgets import QGraphicsWidget, QGridLayout, QWidget, QGraphicsLinearLayout
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtCore import Signal, QPoint
# for test
import sys
sys.path.append("D:\projects\python\MSST-WebUI")

from qfluentwidgets import ConfigValidator, BodyLabel, CheckBox, setTheme, Theme
from ComfyUI.Editor.component.node_port import InputPort, OutputPort

class Node(QGraphicsWidget):
    """节点类"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsWidget.ItemIsMovable)
        self.setFlag(QGraphicsWidget.ItemIsSelectable)
        self.setFlag(QGraphicsWidget.ItemSendsGeometryChanges)
        self.setCacheMode(QGraphicsWidget.DeviceCoordinateCache)
        self.setZValue(-1) 
        self.setPreferredSize(200, 50)
        self.setupUI()
        
    def setupUI(self):
        self.layout = QGraphicsLinearLayout(Qt.Vertical)
        self.setLayout(self.layout)
        self.iopanel = IOPanel(input_list=["input1"], output_list=["output1", "output2, output3"])
        self.parameter_panel = ParameterPanel(parameter_dict=[
        {
            "parameter_name": "aaa",
            "max_value": 100,
            "min_value": 0,
            "default_value": 50,
            "type": int
        },
        {
            "parameter_name": "bbb",
            "max_value": 1.0,
            "min_value": 0.0,
            "default_value": 0.5,
            "type": float
        },
        {
            "parameter_name": "ccc",
            "max_value": True,
            "min_value": False,
            "default_value": True,
            "type": bool
        }
        ])
        self.layout.addItem(self.iopanel)
        self.layout.addItem(self.parameter_panel)
        
    def paint(self, painter, option, widget = None):
        super().paint(painter, option, widget)
    

class IOPanel(QWidget):
    """输入输出面板类"""
    def __init__(self, parent=None, input_list=["input"], output_list=["output"]):
        super().__init__(parent)
        self.input_list = input_list
        self.output_list = output_list
        self.input_ports = []
        self.output_ports = []
        self.setupUI()
        
    def setupUI(self):
        self.grid_layout = QGridLayout(self)
        for i in range(len(self.input_list)):
            input_port = InputPort(text=self.input_list[i])
            self.input_ports.append(input_port)
            self.grid_layout.addWidget(input_port, i, 0)
            
        for i in range(len(self.output_list)):
            output_port = OutputPort(text=self.output_list[i])
            self.output_ports.append(output_port)
            self.grid_layout.addWidget(output_port, i, 1)
        
        self.setLayout(self.grid_layout)
        
    def get_ports(self):
        return self.input_ports, self.output_ports
    

class ParameterPanel(QWidget):
    """参数面板类"""
    parameter_updated = Signal(str, object)  # 当参数更新时触发

    def __init__(self, parent=None, parameter_dict=[]):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.parameter_dict = parameter_dict
        self.parameter_widgets = {}
        self.addParameter()

        self.parameter_updated.connect(self.updateParameter)

    def addParameter(self):
        """
        添加参数字典，字典中的每一项必须包含以下键：

        parameter_dict: List[Dict[str, Any]]
            每个字典项应包含以下键和值：

            - "parameter_name" (str): 参数的名称，必须是字符串。
            - "max_value" (int | float): 参数的最大值，必须是整数或浮动数。
            - "min_value" (int | float): 参数的最小值，必须是整数或浮动数。
            - "default_value" (int | float): 参数的默认值，必须是整数或浮动数。
            - "type" (type): 参数的类型，必须是 int、float 或 bool 之一。

        例如：
        [
            {
                "parameter_name": "aaa",
                "max_value": 100,
                "min_value": 0,
                "default_value": 50,
                "type": int
            },
            {
                "parameter_name": "bbb",
                "max_value": 1.0,
                "min_value": 0.0,
                "default_value": 0.5,
                "type": float
            },
            {
                "parameter_name": "ccc",
                "max_value": True,
                "min_value": False,
                "default_value": True,
                "type": bool
            }
        ]
        """
        for parameter in self.parameter_dict:
            parameter_name = parameter["parameter_name"]
            # max_value = parameter["max_value"]
            # min_value = parameter["min_value"]
            default_value = parameter["default_value"]
            parameter_type = parameter["type"]
            
            widget = None
            if parameter_type in [int, float]:
                text = f"{parameter_name}: {default_value}"
                label = BodyLabel(text)
                label.setFixedHeight(20)
                self.layout.addWidget(label)
                widget = label  # 保存控件
            elif parameter_type == bool:
                checkbox = CheckBox(parameter_name)
                checkbox.setChecked(default_value)
                checkbox.setFixedHeight(20)
                self.layout.addWidget(checkbox)
                widget = checkbox
            else:
                raise ValueError(f"Unsupported parameter type: {parameter_type}")

            self.parameter_widgets[parameter_name] = widget

    def updateParameter(self, parameter_name, new_value):
        """
        更新特定参数的值。
        如果控件存在，则更新控件的显示。
        用法：
        panel = ParameterPanel(parameter_dict=parameter_dict...)
        panel.parameter_updated.emit("aaa", 60)
        panel.parameter_updated.emit("bbb", 0.8)
        """
        widget = self.parameter_widgets.get(parameter_name)
        if widget:
            if isinstance(widget, BodyLabel):
                widget.setText(f"{parameter_name}: {new_value}")
            elif isinstance(widget, CheckBox):
                widget.setChecked(new_value)        
        
    def toImage(self) -> QPixmap:
        """将 QWidget 渲染为 QPixmap"""
        pixmap = self.grab()
        return pixmap
    
    
         
        
if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QGraphicsScene, QGraphicsView
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtCore import Qt
    app = QApplication(sys.argv)
    

    scene = QGraphicsScene()
    view = QGraphicsView(scene)

    # 创建自定义 QGraphicsWidget
    widget = Node()
    scene.addItem(widget)

    # 设置场景和视图
    view.setScene(scene)
    view.show()

    app.exec()
    
    app.exec()