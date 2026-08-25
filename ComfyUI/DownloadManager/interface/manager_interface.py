import os
import json
import hashlib
from PySide6.QtWidgets import QApplication, QAbstractItemView, QFrame, QTableWidgetItem, QHeaderView, QVBoxLayout
from PySide6.QtCore import Qt
from qfluentwidgets import InfoBarPosition, TableWidget, PushButton, IndeterminateProgressRing, Dialog, TitleLabel, CommandBar, Action
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.DownloadManager.common.accessibility import accessible_info_bar, add_accessible_action, configure_command_bar, set_accessible
from ComfyUI.DownloadManager.common.data import models_info


class ManagerInterface(QFrame):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setObjectName("ManagerInterface")
		self.table_data = models_info
		# print("table_data: ", self.table_data)
		self.setupUI()

	def setupUI(self):
		self.layout = QVBoxLayout(self)

		self.settingLabel = TitleLabel(self.tr("Local Model Library"), self)
		self.settingLabel.setFixedHeight(40)
		set_accessible(self.settingLabel, self.tr("Local Model Library"))

		self.command_bar = CommandBar(self)
		self.command_bar.setAccessibleName(self.tr("Local Model Library"))
		self.command_bar.addWidget(self.settingLabel)
		self.command_bar.addSeparator()
		self.refresh_action = Action(FIF.SYNC, self.tr("Refresh"), triggered=self.populateTable)
		add_accessible_action(self.command_bar, self.refresh_action)
		configure_command_bar(self.command_bar, self.tr("More actions"))

		self.layout.addWidget(self.command_bar)

		self.table = TableWidget(self)
		self.table.setBorderVisible(True)
		self.table.setBorderRadius(8)
		self.table.verticalHeader().hide()
		self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
		self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
		self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
		self.table.setTabKeyNavigation(True)
		set_accessible(
			self.table,
			self.tr("Local Model Library"),
			self.tr("Use the arrow keys to browse model rows. Tab moves to the action buttons in the selected row."),
			focusable=True,
		)
		self.table.setColumnCount(5)
		self.table.setHorizontalHeaderLabels([self.tr("model_name"), self.tr("model_class"), self.tr("isInstalled"), self.tr("hashCheck"), self.tr("delete")])
		self.populateTable()

		self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
		self.table.horizontalHeader().resizeSection(1, 150)
		self.table.horizontalHeader().resizeSection(3, 100)
		self.table.horizontalHeader().resizeSection(4, 100)

		self.layout.addWidget(self.table)
		self.layout.setStretch(1, 1)

		self.setLayout(self.layout)

	def populateTable(self):
		self.table.clearContents()
		self.table.setRowCount(len(self.table_data))
		index = 0
		dump = False
		for model in self.table_data:
			model_tab_widget = QTableWidgetItem(model)
			model_tab_widget.setFlags(model_tab_widget.flags() & ~Qt.ItemIsEditable)
			model_tab_widget.setData(Qt.ItemDataRole.AccessibleTextRole, model)
			self.table.setItem(index, 0, model_tab_widget)

			row = self.table_data[model]
			model_class = row["model_class"]
			model_class_tab_widget = QTableWidgetItem(model_class)
			model_class_tab_widget.setFlags(model_class_tab_widget.flags() & ~Qt.ItemIsEditable)
			model_class_tab_widget.setData(Qt.ItemDataRole.AccessibleTextRole, model_class)
			self.table.setItem(index, 1, model_class_tab_widget)

			is_installed = os.path.exists(row["target_position"])
			if self.table_data[model]["is_installed"] != is_installed:
				self.table_data[model]["is_installed"] = is_installed
				dump = True
			installed_text = self.tr("Installed") if is_installed else self.tr("Not installed")
			installed_item = QTableWidgetItem(installed_text)
			installed_item.setCheckState(Qt.Checked if is_installed else Qt.Unchecked)
			installed_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
			installed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
			installed_item.setData(Qt.ItemDataRole.AccessibleTextRole, f"{model}: {installed_text}")
			self.table.setItem(index, 2, installed_item)

			hash_check_button = PushButton(FIF.CERTIFICATE, self.tr("hashCheck"))
			hash_accessible_name = f"{self.tr('hashCheck')}: {model}"
			set_accessible(hash_check_button, hash_accessible_name, tooltip=hash_accessible_name, focusable=True)
			hash_check_button.clicked.connect(lambda checked, model=model: self.hashCheck(model))
			self.table.setCellWidget(index, 3, hash_check_button)

			delete_button = PushButton(FIF.DELETE, self.tr("delete"))
			delete_accessible_name = f"{self.tr('Delete Model')}: {model}"
			set_accessible(delete_button, delete_accessible_name, tooltip=delete_accessible_name, focusable=True)
			delete_button.clicked.connect(lambda checked, model=model: self.deleteModel(model))
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
		checking_text = self.tr("Checking file hash...")
		set_accessible(spinner, checking_text)
		hash_infobar = accessible_info_bar(
			"info",
			title="",
			content=checking_text,
			orient=Qt.Horizontal,
			isClosable=True,
			position=InfoBarPosition.TOP,
			duration=-1,
			parent=self,
			close_text=self.tr("Close"),
		)
		hash_infobar.hBoxLayout.insertWidget(0, spinner)
		hash_infobar.setCustomBackgroundColor("dark", "#39c5bbff")
		QApplication.processEvents()
		row = self.table_data[model]
		file_path = row["target_position"]
		print(os.path.abspath(file_path))
		size = row["model_size"]
		sha256 = row["sha256"]

		if not os.path.exists(file_path):
			accessible_info_bar(
				"error",
				title=self.tr("Hash Check Failed"),
				content=self.tr("File not found"),
				isClosable=True,
				position=InfoBarPosition.TOP,
				duration=5000,
				parent=self,
				close_text=self.tr("Close"),
			)
			hash_infobar.close()
			return

		sz = os.path.getsize(file_path)
		if sz != size:
			accessible_info_bar(
				"error",
				title=self.tr("Hash Check Failed"),
				content=self.tr("File size not match"),
				isClosable=True,
				position=InfoBarPosition.TOP,
				duration=5000,
				parent=self,
				close_text=self.tr("Close"),
			)
			hash_infobar.close()
			return

		hash = self.calculate_sha256(file_path)
		if hash == sha256:
			accessible_info_bar(
				"success",
				title=self.tr("Hash Check Passed"),
				content=self.tr("Hash match"),
				isClosable=True,
				position=InfoBarPosition.TOP,
				duration=5000,
				parent=self,
				close_text=self.tr("Close"),
			)
			hash_infobar.close()
		else:
			accessible_info_bar(
				"error",
				title=self.tr("Hash Check Failed"),
				content=self.tr("Hash not match"),
				isClosable=True,
				position=InfoBarPosition.TOP,
				duration=5000,
				parent=self,
				close_text=self.tr("Close"),
			)
			hash_infobar.close()

	def deleteModel(self, model):
		file_path = self.table_data[model]["target_position"]
		if not os.path.exists(file_path):
			accessible_info_bar(
				"error",
				title=self.tr("Deletion failed"),
				content=self.tr("File not found"),
				isClosable=True,
				position=InfoBarPosition.TOP,
				duration=5000,
				parent=self,
				close_text=self.tr("Close"),
			)
			return

		delete_title = self.tr("Delete Model")
		delete_content = self.tr("Are you sure to delete the model {model}?").format(model=model)
		delete_dialog = Dialog(title=delete_title, content=delete_content, parent=self)
		set_accessible(delete_dialog, delete_title, delete_content)
		if delete_dialog.exec():
			os.remove(file_path)
			# self.table_data[model]['is_installed'] = False
			self.populateTable()
			# self.table.update()
			accessible_info_bar(
				"success",
				title=self.tr("Deletion success"),
				content=self.tr("Model has been deleted successfully"),
				isClosable=True,
				position=InfoBarPosition.TOP,
				duration=5000,
				parent=self,
				close_text=self.tr("Close"),
			)
