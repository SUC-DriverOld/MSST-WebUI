import os
import re
import logging
from PySide6.QtCore import Qt, QObject, Signal, QTimer, QUrl, QFileSystemWatcher
from PySide6.QtWidgets import QFrame, QVBoxLayout
from PySide6.QtGui import QDesktopServices
from qfluentwidgets import (CommandBar, Action, FluentIcon, TextBrowser, 
                         isDarkTheme)
from ComfyUI.Editor.common.logger import GlobalLoggerManager
from ComfyUI.Editor.common.config import font

class LogSignals(QObject):
   new_log = Signal(str, str)  # (log_text, level_name)

class LogInterface(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('LogInterface')
        self._update_enabled = False
        
        self.signals = LogSignals()
        self._current_level = logging.DEBUG
        self.log_file = None
        self.log_buffer = []
        self.last_position = 0
        
        self.setup_ui()
        self.setup_connections()

        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self.on_file_changed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_log_file)
        
        # 连接到GlobalLoggerManager的信号
        manager = GlobalLoggerManager.get_instance()
        manager.log_file_changed.connect(self.on_log_file_changed)
        
        # 创建定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_log_file)
        
        # 检查是否已有日志文件
        if manager.get_log_file():
            self.on_log_file_changed(manager.get_log_file())

    def on_file_changed(self):
        """文件发生变化时的处理"""
        # 确保文件仍在监视列表中
        if self._update_enabled and self.log_file and os.path.exists(self.log_file):
            if self.log_file not in self.watcher.files():
                self.watcher.addPath(self.log_file)
            self.check_log_file()        

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Command Bar for log level selection
        self.commandBar = CommandBar()
        self.commandBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        # Add actions for different log levels
        self.level_actions = {
            'DEBUG': Action(FluentIcon.DEVELOPER_TOOLS, 'Debug'),
            'INFO': Action(FluentIcon.INFO, 'Info'),
            'WARNING': Action(FluentIcon.MEGAPHONE, 'Warning'),
            'ERROR': Action(FluentIcon.CANCEL_MEDIUM, 'Error'),
            'CRITICAL': Action(FluentIcon.QUESTION, 'Critical')
        }
        
        for action in self.level_actions.values():
            self.commandBar.addAction(action)
            
        self.commandBar.addSeparator()
        self.commandBar.addAction(Action(FluentIcon.DOCUMENT, 'Open Log File', triggered=self.open_log_file))
        self.commandBar.addAction(Action(FluentIcon.BRUSH, 'Clear', triggered=self.clear_log))
            
        # Log display area with larger font
        self.log_display = TextBrowser()
        
        # 设置字体
        font.setPointSize(12)
        self.log_display.setFont(font)
        
        layout.addWidget(self.commandBar)
        layout.addWidget(self.log_display)

    def open_log_file(self):
        if self.log_file:
            folder = os.path.dirname(self.log_file)
        else:
            folder = "./logs/ComfyUI"
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def clear_log(self):
        self.log_display.clear()
        self.log_buffer.clear()
        self.last_position = 0

    def setup_connections(self):
        # 连接日志级别选择动作
        self.level_actions['DEBUG'].triggered.connect(
            lambda: self.set_log_level(logging.DEBUG))
        self.level_actions['INFO'].triggered.connect(
            lambda: self.set_log_level(logging.INFO))
        self.level_actions['WARNING'].triggered.connect(
            lambda: self.set_log_level(logging.WARNING))
        self.level_actions['ERROR'].triggered.connect(
            lambda: self.set_log_level(logging.ERROR))
        self.level_actions['CRITICAL'].triggered.connect(
            lambda: self.set_log_level(logging.CRITICAL))
            
        # 连接新日志信号
        self.signals.new_log.connect(self.display_log)

    def on_log_file_changed(self, log_file):
        """响应日志文件变化"""
        # 如果之前有监视的文件，先移除
        if self.log_file and self.log_file in self.watcher.files():
            self.watcher.removePath(self.log_file)
            
        self.log_file = log_file
        self.last_position = 0
        
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                f.seek(0, 2)  # 移动到文件末尾
                self.last_position = f.tell()
            
            # 添加新文件到监视器
            if self._update_enabled:
                self.watcher.addPath(log_file)

    def check_log_file(self):
        """检查日志文件的更新"""
        if not self._update_enabled or not self.log_file or not os.path.exists(self.log_file):
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                f.seek(self.last_position)
                while True:
                    line = f.readline()
                    if not line:
                        break
                        
                    if line.strip():
                        level_match = re.search(r'\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]', line)
                        if level_match:
                            level = level_match.group(1)
                            self.log_buffer.append({'text': line.strip(), 'level': level})
                            self.signals.new_log.emit(line.strip(), level)
                    
                self.last_position = f.tell()
        except Exception as e:
            print(f"Error reading log file: {e}")

    def set_log_level(self, level):
        """设置日志级别并重新显示日志"""
        self._current_level = level
        self.log_display.clear()
        # 重新显示所有符合级别的日志
        for log in self.log_buffer:
            self.display_log(log['text'], log['level'])

    def get_level_color(self, level):
        """获取日志级别对应的颜色"""
        colors = {
            'DEBUG': ('#1E90FF', '#87CEFA'),    # 深蓝/浅蓝
            'INFO': ('#32CD32', '#90EE90'),     # 深绿/浅绿
            'WARNING': ('#FFA500', '#FFD700'),  # 橙色/金色
            'ERROR': ('#FF4500', '#FF6347'),    # 深红/浅红
            'CRITICAL': ('#800080', '#DA70D6')  # 深紫/浅紫
        }
        is_dark = isDarkTheme()
        return colors.get(level, ('white', 'black'))[0 if is_dark else 1]

    def display_log(self, text, level_name):
        """显示日志"""
        # 检查日志级别
        level_num = getattr(logging, level_name)
        if level_num < self._current_level:
            return

        # 用HTML标签处理级别着色
        level_pattern = f"\\[({level_name})\\]"
        color = self.get_level_color(level_name)
        
        # 使用HTML替换，只给级别名称加上颜色
        colored_text = re.sub(
            level_pattern,
            lambda m: f'[<span style="color: {color}">{m.group(1)}</span>]',
            text
        )
        
        # 在末尾添加换行
        self.log_display.append(colored_text)
        
        # 滚动到底部
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    def setUpdatesEnabled(self, enabled):
        """启用或禁用更新"""
        self._update_enabled = enabled
        super().setUpdatesEnabled(enabled)
        
        if enabled:
            if self.log_file and os.path.exists(self.log_file):
                if self.log_file not in self.watcher.files():
                    self.watcher.addPath(self.log_file)
            self.timer.start(100)  # 使用更短的检查间隔，比如50ms
        else:
            if self.log_file and self.log_file in self.watcher.files():
                self.watcher.removePath(self.log_file)
            self.timer.stop()