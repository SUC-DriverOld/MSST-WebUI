from PySide6.QtCore import QObject, QThread, Signal


class NodeExecutor(QObject):
    node_started = Signal(str)  
    node_finished = Signal(str)
    all_finished = Signal()
    
    def __init__(self):
        super().__init__()
        self.current_worker = None
        self.pending_nodes = []
        
    def add_node(self, node_dict: dict):
        self.pending_nodes.append(node_dict)
        if not self.current_worker:
            self._process_next()
            
    def _process_next(self):
        if not self.pending_nodes:
            self.all_finished.emit()
            return
            
        node_dict = self.pending_nodes.pop(0)
        thread = QThread()
        worker = NodeWorker(node_dict)
        
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda: self._on_node_finished(thread))
        
        self.current_worker = worker
        thread.start()
        
    def _on_node_finished(self, thread):
        thread.quit()
        thread.wait()
        thread.deleteLater()
        self.current_worker = None
        import gc
        gc.collect()
        self._process_next()

class NodeWorker(QObject):
    finished = Signal()
    
    def __init__(self, node_dict):
        super().__init__()
        self.node_dict = node_dict
        
    def run(self):
        
        self.finished.emit()