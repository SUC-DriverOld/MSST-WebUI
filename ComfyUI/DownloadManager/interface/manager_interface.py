import os
import json
import hashlib
from PySide6.QtWidgets import QWidget, QHBoxLayout, QFrame, QTableWidgetItem, QHeaderView, QVBoxLayout
from PySide6.QtCore import Qt
from qfluentwidgets import ScrollArea, InfoBar, InfoBarPosition, TableWidget, CheckBox, PushButton, IndeterminateProgressRing, Dialog, TitleLabel, CommandBar, Action
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.DownloadManager.common.data import models_info


class ManagerInterface(QFrame):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setObjectName("ManagerInterface")
		self.table_data = models_info
		self.setupUI()

	def setupUI(self):
		self.layout = QVBoxLayout(self)

		self.settingLabel = TitleLabel(self.tr("Local Model Library"), self)
		self.settingLabel.setFixedHeight(40)

		self.command_bar = CommandBar(self)
		self.command_bar.addWidget(self.settingLabel)
		self.command_bar.addSeparator()
		self.command_bar.addAction(Action(FIF.SYNC, self.tr("Refresh"), triggered=self.populateTable))

		self.layout.addWidget(self.command_bar)

		self.table = TableWidget(self)
		self.table.setBorderVisible(True)
		self.table.setBorderRadius(8)
		self.table.verticalHeader().hide()
		self.table.setColumnCount(5)
		self.table.setHorizontalHeaderLabels([self.tr("model_name"), self.tr("model_class"), self.tr("isInstalled"), self.tr("hashCheck"), self.tr("delete")])
		self.populateTable()

		self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
		self.table.horizontalHeader().resizeSection(1, 150)
		self.table.horizontalHeader().resizeSection(3, 100)
		self.table.horizontalHeader().resizeSection(4, 100)

		self.scroll_area = ScrollArea(self)
		self.scroll_area.setStyleSheet("background-color: transparent;")
		self.scroll_area.setWidgetResizable(True)
		self.scroll_area.setWidget(self.table)

		self.layout.addWidget(self.scroll_area)
		self.layout.setStretch(1, 1)

		self.setLayout(self.layout)

	def populateTable(self):
		self.table.setRowCount(len(self.table_data))
		index = 0
		dump = False
		for model in self.table_data:
			model_tab_widget = QTableWidgetItem(model)
			model_tab_widget.setFlags(model_tab_widget.flags() & ~Qt.ItemIsEditable)
			self.table.setItem(index, 0, model_tab_widget)

			row = self.table_data[model]
			model_class = row["model_class"]
			model_class_tab_widget = QTableWidgetItem(model_class)
			model_class_tab_widget.setFlags(model_class_tab_widget.flags() & ~Qt.ItemIsEditable)
			self.table.setItem(index, 1, model_class_tab_widget)

			# is_installed = row['is_installed']
			checkbox = CheckBox()
			is_installed = os.path.exists(row["target_position"])
			if self.table_data[model]["is_installed"] != is_installed:
				self.table_data[model]["is_installed"] = is_installed
				dump = True
			checkbox.setChecked(is_installed)
			checkbox.setEnabled(False)
			checkbox_layout = QHBoxLayout()
			checkbox_layout.setAlignment(Qt.AlignCenter)
			checkbox_layout.addWidget(checkbox)
			checkbox_widget = QWidget()
			checkbox_widget.setLayout(checkbox_layout)
			self.table.setCellWidget(index, 2, checkbox_widget)

			hash_check_button = PushButton(FIF.CERTIFICATE, "Sha256")
			hash_check_button.clicked.connect(lambda checked, index=index, model=model: self.hashCheck(model))
			self.table.setCellWidget(index, 3, hash_check_button)

			delete_button = PushButton(FIF.DELETE, "Delete")
			delete_button.clicked.connect(lambda checked, index=index, model=model: self.deleteModel(model))
			self.table.setCellWidget(index, 4, delete_button)

			index += 1

		if dump:
			with open("./data/models_info.json", "w") as f:
				json.dump(self.table_data, f, indent=4)

	def calculate_sha256(self, file_path):
		sha256_hash = hashlib.sha256()
		with open(file_path, "rb") as f:
			for byte_block in iter(lambda: f.read(4096), b""):
				sha256_hash.update(byte_block)
		return sha256_hash.hexdigest()

	def hashCheck(self, model):
		spinner = IndeterminateProgressRing(self)
		spinner.setFixedSize(15, 15)
		spinner.setStrokeWidth(3)
		hash_infobar = InfoBar.info(title="", content="hash校验中...", orient=Qt.Horizontal, isClosable=True, position=InfoBarPosition.TOP, duration=-1, parent=self)
		hash_infobar.hBoxLayout.insertWidget(0, spinner)
		hash_infobar.setCustomBackgroundColor("dark", "#39c5bbff")
		row = self.table_data[model]
		file_path = row["target_position"]
		print(os.path.abspath(file_path))
		size = row["model_size"]
		sha256 = row["sha256"]

		if not os.path.exists(file_path):
			InfoBar.error(title=self.tr("Hash Check Failed"), content=self.tr("File not found"), isClosable=True, position=InfoBarPosition.TOP, duration=5000, parent=self)
			hash_infobar.close()
			return

		sz = os.path.getsize(file_path)
		if sz != size:
			InfoBar.error(title=self.tr("Hash Check Failed"), content=self.tr("File size not match"), isClosable=True, position=InfoBarPosition.TOP, duration=5000, parent=self)
			hash_infobar.close()
			return

		hash = self.calculate_sha256(file_path)
		if hash == sha256:
			InfoBar.success(title=self.tr("Hash Check Passed"), content=self.tr("Hash match"), isClosable=True, position=InfoBarPosition.TOP, duration=5000, parent=self)
			hash_infobar.close()
		else:
			InfoBar.error(title=self.tr("Hash Check Failed"), content=self.tr("Hash not match"), isClosable=True, position=InfoBarPosition.TOP, duration=5000, parent=self)
			hash_infobar.close()

	def deleteModel(self, model):
		file_path = self.table_data[model]["target_position"]
		if not os.path.exists(file_path):
			InfoBar.error(title=self.tr("Deletion failed"), content=self.tr("File not found"), isClosable=True, position=InfoBarPosition.TOP, duration=5000, parent=self)
			return

		delete_dialog = Dialog(title=self.tr("Delete Model"), content=self.tr(f"Are you sure to delete the model {model}?"), parent=self)
		if delete_dialog.exec():
			os.remove(file_path)
			# self.table_data[model]['is_installed'] = False
			self.populateTable()
			# self.table.update()
			InfoBar.success(title=self.tr("Deletion success"), content=self.tr("Model has been deleted successfully"), isClosable=True, position=InfoBarPosition.TOP, duration=5000, parent=self)
