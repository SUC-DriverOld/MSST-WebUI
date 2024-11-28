from PySide6.QtCore import QThread, Signal

class NodeExecutor(QThread):
    execute_finished = Signal()
    execute_error = Signal(str)
    running_node = Signal(str)
    node_finished = Signal(str)
    
    def __init__(self, node_dict_list=[], parent=None):
        super().__init__(parent)
        self.node_dict_list = node_dict_list
        
    def add_node_dict(self, node_dict):
        self.node_dict_list.append(node_dict)
        
    def run(self):
        for node_dict in self.node_dict_list:
            print(node_dict)
        self.execute_finished.emit()