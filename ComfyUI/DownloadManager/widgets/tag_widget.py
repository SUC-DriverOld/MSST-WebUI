from PySide6.QtWidgets import QWidget, QHBoxLayout, QSizePolicy
from PySide6.QtCore import Signal
from qfluentwidgets import ToolButton, BodyLabel, FluentIcon as FIF
from ComfyUI.DownloadManager.common.accessibility import set_accessible


class TagWidget(QWidget):
	deleteSignal = Signal(str)

	def __init__(self, text: str, remove_text: str, parent=None):
		super().__init__(parent)

		self.text = text
		self.remove_text = remove_text
		self.setupUI()

	def setupUI(self):
		layout = QHBoxLayout(self)
		layout.setContentsMargins(5, 5, 5, 5)
		self.text_label = BodyLabel(self)
		self.text_label.setText(self.text)
		set_accessible(self, self.text)
		set_accessible(self.text_label, self.text)
		# text_label.setFont(QFont("Consolas", 12))
		layout.addWidget(self.text_label)

		self.delete_button = ToolButton(FIF.UNPIN)
		self.delete_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
		self.delete_button.setFixedSize(28, 28)
		self.delete_button.setStyleSheet("background-color: transparent")
		accessible_name = f"{self.remove_text}: {self.text}"
		set_accessible(self.delete_button, accessible_name, tooltip=accessible_name, focusable=True)

		self.delete_button.clicked.connect(self.onDeleteClicked)
		layout.addWidget(self.delete_button)
		self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
		self.setFixedHeight(36)
		# self.setStyleSheet("""
		#     background-color: rgba(169, 169, 169, 0.9);
		#     border-radius: 15px;
		#     padding: 5px 10px;
		# """)

	def onDeleteClicked(self):
		self.deleteSignal.emit(self.text)
