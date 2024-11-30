from datetime import datetime
import logging
import os
from colorama import Fore, Style, init
from PySide6.QtCore import Signal, QObject

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
        # 使用 logging.getLogger 获取或创建 logger
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
        
        # 确保 logger 不会传播到父 logger
        logger.propagate = False
        
        # 将 logger 添加到全局管理器
        # GlobalLoggerManager.get_instance()._loggers = getattr(
        #     GlobalLoggerManager.get_instance(), '_loggers', []
        # )
        
        return logger
        
def create_queue_handler(log_queue):
    """创建队列处理器用于GUI显示"""
    handler = logging.Handler()
    handler.emit = lambda record: log_queue.put(handler.format(record))
    return handler

class GlobalLoggerManager(QObject):
    logger_updated = Signal(object)  # 用于通知 logger 的添加或更新
    log_file_changed = Signal(str)   # 用于通知日志文件路径的更改
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = QObject.__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            super().__init__()
            self._initialized = True
            self._main_logger = None
            self._loggers = []
            self._log_file = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def add_logger(cls, logger, is_main=False, log_file=None):
        """统一的logger添加/更新方法
        
        Args:
            logger: 要添加的logger
            is_main: 是否为主logger
            log_file: 日志文件路径（仅当is_main=True时需要）
        """
        instance = cls.get_instance()
        
        if is_main:
            instance._main_logger = logger
            if log_file:
                instance._log_file = log_file
                print("Log file changed:", log_file)
                instance.log_file_changed.emit(log_file)
            
        if logger not in instance._loggers:
            instance._loggers.append(logger)
            
        instance.logger_updated.emit(logger)
    
    @classmethod
    def get_main_logger(cls):
        return cls.get_instance()._main_logger
    
    @classmethod
    def get_all_loggers(cls):
        return cls.get_instance()._loggers
    
    @classmethod
    def get_log_file(cls):
        return cls.get_instance()._log_file
    
    @classmethod
    def add_debug_log(cls, log_text):
        cls.get_main_logger().debug(log_text)

    @classmethod
    def add_info_log(cls, log_text):
        cls.get_main_logger().info(log_text)

    @classmethod
    def add_warning_log(cls, log_text):
        cls.get_main_logger().warning(log_text)

    @classmethod
    def add_error_log(cls, log_text):
        cls.get_main_logger().error(log_text)

    @classmethod
    def add_critical_log(cls, log_text):
        cls.get_main_logger().critical(log_text)