from PySide6.QtCore import QThread, Signal
import torch.multiprocessing as mp
import traceback
from ComfyUI.Editor.common.logger import LoggerFactory, GlobalLoggerManager

class InferenceProcess(mp.Process):
    def __init__(self, task_queue, result_queue, msst_inference_func, vr_inference_func, log_path):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.msst_inference_func = msst_inference_func
        self.vr_inference_func = vr_inference_func
        self.log_path = log_path
        
    def run(self):
        log_factory = LoggerFactory()
        logger = log_factory.get_process_logger(self.log_path, f"Process-{self.pid}")
        GlobalLoggerManager.add_logger(logger)
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    break
                
                node_dict = task
                if node_dict["model_type"] is not None:
                    self.msst_inference_func(node_dict, logger)
                else:
                    self.vr_inference_func(node_dict, logger)
                    
                self.result_queue.put(('success', None))
                    
            except Exception as e:
                logger.error(f"Process error: {str(e)}\n{traceback.format_exc()}")
                self.result_queue.put(('error', f"Process error: {str(e)}\n{traceback.format_exc()}"))
                
        self.result_queue.put(('finish', None))

class InferenceWorker(QThread):
    progress_signal = Signal(str)
    result_signal = Signal(dict)
    error_signal = Signal(str)
    finished_signal = Signal()
    
    def __init__(self, nodes_to_run, msst_inference_func, vr_inference_func, log_path):
        super().__init__()
        self.nodes_to_run = nodes_to_run
        self.msst_inference_func = msst_inference_func
        self.vr_inference_func = vr_inference_func
        self.log_path = log_path
        self.is_running = True
        
    def run(self):
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        process = None
        
        try:
            process = InferenceProcess(
                task_queue, 
                result_queue, 
                self.msst_inference_func,
                self.vr_inference_func,
                self.log_path
            )
            process.start()
            
            for node in self.nodes_to_run:
                if not self.is_running:
                    break
                
                node_type = "msst node" if node.node_dict["model_type"] else "vr node"
                self.progress_signal.emit(f"Processing {node_type}: {node.node_dict.get('model_name', 'Unknown')}")
                
                task_queue.put(node.node_dict)
                
                status, data = result_queue.get()
                if status == 'error':
                    self.error_signal.emit(str(data))
                    break
                elif status == 'success':
                    self.result_signal.emit(node.node_dict)
            
            if self.is_running:
                self.finished_signal.emit()
                
        except Exception as e:
            self.error_signal.emit(f"Worker error: {str(e)}\n{traceback.format_exc()}")
        finally:
            if process and process.is_alive():
                task_queue.put(None)
                process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
            
            task_queue.close()
            result_queue.close()
    
    def stop(self):
        self.is_running = False