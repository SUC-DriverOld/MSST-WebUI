from PySide6.QtCore import QTimer, Qt, QDateTime
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QWidget
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
from PySide6.QtGui import QPen, QColor, QPainter, QBrush
from qfluentwidgets import (ProgressRing, ScrollArea, SimpleCardWidget, SubtitleLabel, 
                          StrongBodyLabel, BodyLabel)
import psutil
import GPUtil
import platform
import cpuinfo
from datetime import datetime
from collections import deque
from ComfyUI.Editor.common.config import cfg

class SystemInfoCard(SimpleCardWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setup_ui(title)
    
    def setup_ui(self, title):
        layout = QVBoxLayout(self)
        self.title_label = SubtitleLabel(title)
        self.info_layout = QVBoxLayout()
        
        layout.addWidget(self.title_label)
        layout.addLayout(self.info_layout)
        
    def add_info_item(self, label, value):
        item_layout = QHBoxLayout()
        label_widget = StrongBodyLabel(f"{label}:")
        value_widget = BodyLabel(str(value))
        item_layout.addWidget(label_widget)
        item_layout.addWidget(value_widget)
        item_layout.addStretch()
        self.info_layout.addLayout(item_layout)

class MonitorCard(SimpleCardWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setup_ui(title)
    
    def setup_ui(self, title):
        layout = QVBoxLayout(self)
        
        self.title_label = SubtitleLabel(title)
        
        self.ring = ProgressRing()
        self.ring.setFixedSize(120, 120)
        self.ring.setRange(0, 100)
        self.ring.setValue(0)
        self.ring.setTextVisible(False)  # 关闭环内文字
        self.ring.setStrokeWidth(4)
        
        # 添加环下方的数值显示
        self.value_label = SubtitleLabel("0")  # 使用SubtitleLabel让数字更醒目
        self.value_label.setAlignment(Qt.AlignCenter)
        
        self.detail_layout = QVBoxLayout()
        self.primary_detail = BodyLabel("--")
        self.secondary_detail = BodyLabel("--")
        self.detail_layout.addWidget(self.primary_detail)
        self.detail_layout.addWidget(self.secondary_detail)
        
        layout.addWidget(self.title_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.ring, alignment=Qt.AlignCenter)
        layout.addWidget(self.value_label)
        layout.addLayout(self.detail_layout)
        layout.setSpacing(10)

class PerformanceChart(SimpleCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.data_points = 60
        self.setup_series()
        
        # 监听主题变化
        cfg.themeChanged.connect(self.update_chart_theme)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.title_label = SubtitleLabel("Performance History")
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle("Resource Usage History")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        
        # 创建图表视图
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setMinimumHeight(300)  # 设置最小高度确保图表可见
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.chart_view)
        
    def get_theme_colors(self):
        """根据当前主题返回合适的颜色"""
        theme = cfg.get(cfg.theme)
        if theme == "Light":
            return {
                'cpu': "#FF6B6B",      # 红色
                'ram': "#4ECDC4",      # 青色
                'gpu': "#45B7D1",      # 蓝色
                'vram': "#FFB400",     # 金色
                'background': "#FFFFFF",# 白色背景
                'text': "#000000",     # 黑色文字
                'grid': "#E5E5E5"      # 浅灰色网格
            }
        else:  # Dark or Auto in dark mode
            return {
                'cpu': "#FF8585",      # 亮红色
                'ram': "#6FE7DF",      # 亮青色
                'gpu': "#65D7F1",      # 亮蓝色
                'vram': "#FFD54F",     # 亮金色
                'background': "#2B2B2B",# 深色背景
                'text': "#FFFFFF",     # 白色文字
                'grid': "#404040"      # 深灰色网格
            }
    
    def update_chart_theme(self):
        """更新图表主题"""
        colors = self.get_theme_colors()
        
        # 更新线条颜色
        self.cpu_series.setPen(QPen(QColor(colors['cpu']), 2))
        self.ram_series.setPen(QPen(QColor(colors['ram']), 2))
        self.gpu_series.setPen(QPen(QColor(colors['gpu']), 2))
        self.vram_series.setPen(QPen(QColor(colors['vram']), 2))
        
        # 更新图表主题
        self.chart.setBackgroundBrush(QBrush(QColor(colors['background'])))
        self.chart.legend().setLabelColor(QColor(colors['text']))
        
        # 更新坐标轴颜色
        self.axis_x.setLabelsColor(QColor(colors['text']))
        self.axis_y.setLabelsColor(QColor(colors['text']))
        self.axis_x.setLinePen(QPen(QColor(colors['grid'])))
        self.axis_y.setLinePen(QPen(QColor(colors['grid'])))
        self.axis_x.setGridLinePen(QPen(QColor(colors['grid'])))
        self.axis_y.setGridLinePen(QPen(QColor(colors['grid'])))
        
        # 更新标题颜色
        self.chart.setTitleBrush(QBrush(QColor(colors['text'])))
    
    def setup_series(self):
        # 创建数据系列
        self.cpu_series = QLineSeries()
        self.ram_series = QLineSeries()
        self.gpu_series = QLineSeries()
        self.vram_series = QLineSeries()
        
        self.cpu_series.setName("CPU")
        self.ram_series.setName("Memory")
        self.gpu_series.setName("GPU")
        self.vram_series.setName("VRAM")
        
        # 使用主题颜色
        colors = self.get_theme_colors()
        self.cpu_series.setPen(QPen(QColor(colors['cpu']), 2))
        self.ram_series.setPen(QPen(QColor(colors['ram']), 2))
        self.gpu_series.setPen(QPen(QColor(colors['gpu']), 2))
        self.vram_series.setPen(QPen(QColor(colors['vram']), 2))
        
        # 添加系列到图表
        self.chart.addSeries(self.cpu_series)
        self.chart.addSeries(self.ram_series)
        self.chart.addSeries(self.gpu_series)
        self.chart.addSeries(self.vram_series)
        
        # 设置坐标轴和图表主题
        self.setup_axes()
        self.update_chart_theme()
        
        # 初始化数据存储
        self.timestamps = deque(maxlen=self.data_points)
        self.cpu_data = deque(maxlen=self.data_points)
        self.ram_data = deque(maxlen=self.data_points)
        self.gpu_data = deque(maxlen=self.data_points)
        self.vram_data = deque(maxlen=self.data_points)
    
    def setup_axes(self):
        self.axis_x = QDateTimeAxis()
        self.axis_x.setFormat("mm:ss")
        self.axis_x.setTitleText("Time")
        
        self.axis_y = QValueAxis()
        self.axis_y.setRange(0, 100)
        self.axis_y.setTitleText("Usage %")
        
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        for series in [self.cpu_series, self.ram_series, 
                      self.gpu_series, self.vram_series]:
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)

    def update_chart(self, cpu_percent, ram_percent, gpu_percent, vram_percent):  # 添加vram_percent参数
        timestamp = QDateTime.currentDateTime()
        
        # 更新数据
        self.timestamps.append(timestamp)
        self.cpu_data.append(cpu_percent)
        self.ram_data.append(ram_percent)
        self.gpu_data.append(gpu_percent)
        self.vram_data.append(vram_percent)
        
        # 清除旧数据
        self.cpu_series.clear()
        self.ram_series.clear()
        self.gpu_series.clear()
        self.vram_series.clear()
        
        # 添加新数据
        for i in range(len(self.timestamps)):
            ts = self.timestamps[i].toMSecsSinceEpoch()
            self.cpu_series.append(ts, self.cpu_data[i])
            self.ram_series.append(ts, self.ram_data[i])
            self.gpu_series.append(ts, self.gpu_data[i])
            self.vram_series.append(ts, self.vram_data[i])
        
        # 更新X轴范围
        if self.timestamps:
            self.axis_x.setRange(self.timestamps[0], self.timestamps[-1])        

class ManagerInterface(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('ManagerInterface')
        self._update_enabled = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.collect_system_info()
        self.setup_ui()
        
    def collect_system_info(self):
        self.system_info = {
            "OS": f"{platform.system()} {platform.version()}",
            "Architecture": platform.machine(),
            "Processor": cpuinfo.get_cpu_info()["brand_raw"],
            "CPU Cores": f"{psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count()} Logical",
            "Total RAM": f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB"
        }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.system_info["GPU"] = gpu.name
                self.system_info["GPU Driver"] = gpu.driver
                self.system_info["Total VRAM"] = f"{gpu.memoryTotal} MB"
        except:
            self.system_info["GPU"] = "Not detected"
        
    def setup_ui(self):
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create scroll area
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # Title
        content_layout.addWidget(SubtitleLabel("System Monitor"))
        
        # System Information Section
        sys_info_card = SystemInfoCard("System Information")
        for key, value in self.system_info.items():
            sys_info_card.add_info_item(key, value)
        content_layout.addWidget(sys_info_card)
        
        # Performance Monitoring Cards
        monitors_layout = QHBoxLayout()  # 改用水平布局
    
        self.cpu_card = MonitorCard("CPU Usage")
        self.ram_card = MonitorCard("Memory Usage")
        self.gpu_card = MonitorCard("GPU Usage")
        self.vram_card = MonitorCard("VRAM Usage")
        
        monitors_layout.addWidget(self.cpu_card)
        monitors_layout.addWidget(self.ram_card)
        monitors_layout.addWidget(self.gpu_card)
        monitors_layout.addWidget(self.vram_card)
        
        content_layout.addLayout(monitors_layout)
        
        # Performance Chart
        self.performance_chart = PerformanceChart()
        content_layout.addWidget(self.performance_chart)
        
        # Timestamp
        self.timestamp_label = BodyLabel("Last update: Never")
        self.timestamp_label.setAlignment(Qt.AlignRight)
        content_layout.addWidget(self.timestamp_label)
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        
    def update_stats(self):
        if not self._update_enabled:
            return
            
        # Update CPU stats
        cpu_percent = psutil.cpu_percent()
        self.cpu_card.ring.setValue(int(cpu_percent))
        self.cpu_card.value_label.setText(f"{int(cpu_percent)}%")
        self.cpu_card.ring.setFormat("%v")
        freq = psutil.cpu_freq()
        self.cpu_card.primary_detail.setText(f"Current: {freq.current:.1f} MHz")
        self.cpu_card.secondary_detail.setText(f"Cores usage: {sum(psutil.cpu_percent(percpu=True))/psutil.cpu_count():.1f}%")
        
        # Update RAM stats
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_used = ram.used / (1024**3)
        ram_total = ram.total / (1024**3)
        self.ram_card.ring.setValue(int(ram_percent))
        self.ram_card.value_label.setText(f"{int(ram_percent)}%")
        self.ram_card.ring.setFormat("%v%")
        self.ram_card.primary_detail.setText(f"Used: {ram_used:.1f} GB")
        self.ram_card.secondary_detail.setText(f"Total: {ram_total:.1f} GB")
        
        # GPU stats
        gpu_percent = 0
        vram_percent = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = int(gpu.load * 100)
                self.gpu_card.ring.setValue(gpu_percent)
                self.gpu_card.ring.setFormat("%v%")
                self.gpu_card.value_label.setText(f"{gpu_percent}%")
                self.gpu_card.primary_detail.setText(f"Temp: {gpu.temperature}°C")
                self.gpu_card.secondary_detail.setText(f"Driver: {gpu.driver}W")
                
                vram_used = gpu.memoryUsed
                vram_total = gpu.memoryTotal
                vram_percent = (vram_used / vram_total) * 100
                self.vram_card.ring.setValue(int(vram_percent))
                self.vram_card.ring.setFormat("%v%")
                self.vram_card.value_label.setText(f"{int(vram_percent)}%")
                self.vram_card.primary_detail.setText(f"Used: {vram_used} MB")
                self.vram_card.secondary_detail.setText(f"Total: {vram_total} MB")
        except:
            self.gpu_card.ring.setValue(0)
            self.vram_card.ring.setValue(0)
            self.gpu_card.value_label.setText("0%")
            self.vram_card.value_label.setText("0%")
            self.gpu_card.primary_detail.setText("GPU not detected")
            self.vram_card.primary_detail.setText("VRAM not detected")
        
        # Update performance chart
        self.performance_chart.update_chart(cpu_percent, ram_percent, gpu_percent, vram_percent)
        
        # Update timestamp
        self.timestamp_label.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
        
    def setUpdatesEnabled(self, enabled):
        self._update_enabled = enabled
        super().setUpdatesEnabled(enabled)
        
        if enabled:
            self.timer.start(1000)  # Update every second
        else:
            self.timer.stop()


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import setTheme, Theme
    import sys
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)
    window = ManagerInterface()
    window.showMaximized()
    window.setUpdatesEnabled(True)
    
    sys.exit(app.exec())