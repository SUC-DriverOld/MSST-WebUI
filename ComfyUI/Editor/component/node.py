from PySide6.QtWidgets import QGraphicsWidget, QGridLayout, QWidget
# for test
import sys
sys.path.append("/home/tong/projects/python/MSST-WebUI")
from PySide6.QtWidgets import QApplication

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
        
    def setupUI(self):
        pass    
    

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
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IOPanel(input_list=["input1", "input2", "input3"], output_list=["output1", "output2"])
    window.show()
    sys.exit(app.exec())