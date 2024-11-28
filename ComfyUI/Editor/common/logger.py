from datetime import datetime
import logging
import os
from colorama import Fore, Style, init

class LoggerFactory:
    MAX_LOG = 50
    LOG_DIR = "logs"
    
    def __init__(self):
        init(autoreset=True)
        
    class ColorFormatter(logging.Formatter):
        def format(self, record):
            record.pathname = os.path.relpath(record.pathname)
            log_msg = super().format(record)
            
            color_map = {
                "INFO": Fore.GREEN,
                "DEBUG": Fore.BLUE,
                "WARNING": Fore.YELLOW,
                "ERROR": Fore.RED,
                "CRITICAL": Fore.MAGENTA
            }
            
            if record.levelname in color_map:
                log_msg = log_msg.replace(
                    record.levelname,
                    f"{color_map[record.levelname]}{record.levelname}{Style.RESET_ALL}"
                )
                
            return log_msg
            
    @staticmethod
    def manage_log_files(log_dir, max_log):
        """管理日志文件，保持文件数量在限制之内"""
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        log_files = sorted(
            log_files,
            key=lambda f: datetime.strptime(f.split(".")[0], "%Y-%m-%d")
        )
        
        while len(log_files) > max_log:
            try:
                oldest_file = log_files.pop(0)
                os.remove(os.path.join(log_dir, oldest_file))
            except Exception:
                pass
                
    @staticmethod
    def set_log_level(logger, level):
        """设置日志级别"""
        logger.console_handler.setLevel(level)
        
    def get_logger(self, console_level=logging.INFO, max_log=MAX_LOG):
        """获取配置好的logger实例"""
        logger = logging.getLogger("logger")
        
        if logger.hasHandlers():
            return logger
            
        logger.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        formatter = self.ColorFormatter(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # 确保日志目录存在
        os.makedirs(self.LOG_DIR, exist_ok=True)
        
        # 文件处理器
        log_filename = datetime.now().strftime("%Y-%m-%d.log")
        file_path = os.path.join(self.LOG_DIR, log_filename)
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        
        if not logger.hasHandlers():
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
        self.manage_log_files(self.LOG_DIR, max_log)
        logger.console_handler = console_handler
        
        return logger
        
    def get_run_logger(self, run_dir, console_level=logging.INFO):
        """获取单次运行的logger实例"""
        logger = logging.getLogger(f"run_logger_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        logger.setLevel(logging.DEBUG)
        
        # 运行日志文件处理器
        log_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_path = os.path.join(run_dir, log_filename)
        file_handler = logging.FileHandler(file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(self.ColorFormatter(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        ))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.console_handler = console_handler
        
        return logger
    
    def get_process_logger(self, log_path, process_name):
        """获取进程专用的logger实例，使用同一个日志文件"""
        logger = logging.getLogger(f"process_logger_{process_name}")
        logger.setLevel(logging.DEBUG)
        
        # 如果logger已经有处理器，直接返回
        if logger.handlers:
            return logger
            
        # 文件处理器 - 使用已存在的日志文件
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt=f"%(asctime)s.%(msecs)03d [{process_name}] [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(self.ColorFormatter(
            fmt=f"%(asctime)s.%(msecs)03d [{process_name}] [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        ))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.console_handler = console_handler
        
        return logger
        
def create_queue_handler(log_queue):
    """创建队列处理器用于GUI显示"""
    handler = logging.Handler()
    handler.emit = lambda record: log_queue.put(handler.format(record))
    return handler

class GlobalLoggerManager:
    _instance = None
    _logger = None
    _log_window = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalLoggerManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def set_logger(cls, logger, log_window=None):
        instance = cls.get_instance()
        instance._logger = logger
        instance._log_window = log_window
    
    @classmethod
    def get_logger(cls):
        return cls.get_instance()._logger
        
    @classmethod
    def get_log_window(cls):
        return cls.get_instance()._log_window
    
    @classmethod
    def show_log_window(cls):
        log_window = cls.get_log_window()
        if log_window and not log_window.isVisible():
            log_window.show()