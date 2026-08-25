from PySide6.QtWidgets import QAbstractItemView, QFrame, QWidget, QHBoxLayout, QTreeWidgetItem
from PySide6.QtCore import Qt, QEasingCurve, QUrl
from PySide6.QtGui import QDesktopServices
from qfluentwidgets import CommandBar, FlowLayout, ScrollArea, Action, VBoxLayout, TreeWidget, ProgressBar, BodyLabel, SmoothMode, InfoBarPosition, TitleLabel
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.DownloadManager.common.accessibility import accessible_info_bar, add_accessible_action, configure_command_bar, set_accessible
from huggingface_hub import hf_hub_url
from ComfyUI.DownloadManager.common.data import ARIA2_RPC_URL, HF_ENDPOINT, ARIA2_RPC_SECRET, models_info
from ComfyUI.DownloadManager.common.download_thread import DownloadThread
from ComfyUI.DownloadManager.widgets.tag_widget import TagWidget
import requests
import json
import os


class DownloadInterface(QFrame):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setObjectName("DownloadInterface")
		self.model_to_download = {"multi_stem_models": [], "single_stem_models": [], "vocal_models": [], "VR_Models": []}
		self.model_urls = []
		self.total_progress = 0
		self.total_files = 0
		self.download_speed = 0
		self.total_downloaded = 0
		self.setupUI()

	def setupUI(self):
		self.command_bar = CommandBar(self)
		self.command_bar.setAccessibleName(self.tr("Download Center"))
		self.command_bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

		self.label = TitleLabel(self.tr("Download Center"))
		self.label.setStyleSheet("color: white;")
		set_accessible(self.label, self.tr("Download Center"))

		self.flow_widget = QWidget(self)
		self.flow_layout = FlowLayout(self.flow_widget)
		self.flow_layout.setAnimation(250, QEasingCurve.OutQuad)

		self.flow_layout.setContentsMargins(30, 30, 30, 30)
		self.flow_layout.setVerticalSpacing(20)
		self.flow_layout.setHorizontalSpacing(10)

		scroll_area = ScrollArea(self)
		scroll_area.setWidgetResizable(True)
		scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		scroll_area.setMinimumHeight(200)
		scroll_area.setStyleSheet("QScrollArea{background: transparent; border: none}")

		self.flow_widget.setStyleSheet("QWidget{background: transparent}")
		scroll_area.setWidget(self.flow_widget)

		self.command_bar.addWidget(self.label)
		self.command_bar.addSeparator()
		self.open_folder_action = Action(FIF.FOLDER, self.tr("Open Folder"), triggered=self.openFolder)
		self.download_action = Action(FIF.DOWNLOAD, self.tr("Download Model(s)"), triggered=self.download_all_models)
		self.aria2_action = Action(FIF.SEND, self.tr("Send To Aria2"), triggered=self.send_to_aria2)
		self.clear_action = Action(FIF.DELETE, self.tr("clear"), triggered=self.clearModels)
		add_accessible_action(self.command_bar, self.open_folder_action)
		add_accessible_action(self.command_bar, self.download_action)
		# self.command_bar.addAction(Action(FIF.CANCEL, '取消下载', triggered=self.download_thread.terminate))
		add_accessible_action(self.command_bar, self.aria2_action)
		add_accessible_action(self.command_bar, self.clear_action)
		configure_command_bar(self.command_bar, self.tr("More actions"))
		self.command_bar.addSeparator()
		self.layout = VBoxLayout(self)
		self.layout.addWidget(self.command_bar)

		separator = QFrame(self)
		separator.setFrameShape(QFrame.HLine)
		separator.setFrameShadow(QFrame.Sunken)
		self.layout.addWidget(separator)
		self.layout.addWidget(scroll_area)

		self.populateTree()

		self.layout.addWidget(self.tree)
		# self.layout.setStretch(2, 1)

		self.tree.itemChanged.connect(self.treeItemChanged)

		self.total_progress_bar = ProgressBar(self)
		self.total_progress_bar.setRange(0, 100)
		set_accessible(self.total_progress_bar, self.tr("Total download progress:"))

		self.single_progress_bar = ProgressBar(self)
		self.single_progress_bar.setRange(0, 100)
		set_accessible(self.single_progress_bar, self.tr("Current file download progress:"))

		self.single_progress_label = BodyLabel(self.tr("Current file download progress:"), self)
		self.total_progress_label = BodyLabel(self.tr("Total download progress:"), self)
		self.download_speed_label = BodyLabel(self.tr("No active download task"), self)
		self.download_speed_label.setStyleSheet("background: transparent;")
		set_accessible(self.download_speed_label, self.tr("No active download task"))
		self.single_progress_label.setBuddy(self.single_progress_bar)
		self.total_progress_label.setBuddy(self.total_progress_bar)

		single_progress_layout = QHBoxLayout()
		single_progress_layout.addWidget(self.single_progress_label)
		single_progress_layout.addWidget(self.single_progress_bar)

		total_progress_layout = QHBoxLayout()
		total_progress_layout.addWidget(self.total_progress_label)
		total_progress_layout.addWidget(self.total_progress_bar)

		self.progress_widget = QWidget(self)
		self.progress_widget.setStyleSheet("background-color: transparent;")
		self.progress_widget_layout = VBoxLayout(self.progress_widget)
		self.progress_widget_layout.addLayout(single_progress_layout)
		self.progress_widget_layout.addLayout(total_progress_layout)
		self.progress_widget_layout.addWidget(self.download_speed_label)
		# self.progress_widget.setFixedHeight(120)

		self.layout.addWidget(self.progress_widget)

	def openFolder(self):
		folder_path = "./pretrain"
		QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))

	def populateTree(self):
		self.tree = TreeWidget(self)
		self.tree.setFixedHeight(400)
		self.tree.setHeaderLabel(self.tr("Model Library"))
		self.tree.setBorderVisible(False)
		self.tree.setIndentation(50)
		self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
		self.tree.setAllColumnsShowFocus(True)
		set_accessible(
			self.tree,
			self.tr("Model Library"),
			self.tr("Use the arrow keys to browse models and Space to select or clear a model."),
			focusable=True,
		)
		self.tree.scrollDelagate.verticalSmoothScroll.setSmoothMode(SmoothMode.NO_SMOOTH)

		model_class_item1 = QTreeWidgetItem(["multi_stem_models"])
		model_class_item2 = QTreeWidgetItem(["single_stem_models"])
		model_class_item3 = QTreeWidgetItem(["vocal_models"])
		model_class_item4 = QTreeWidgetItem(["VR_Models"])

		categorized_models = {"multi_stem_models": [], "single_stem_models": [], "vocal_models": [], "VR_Models": []}

		for model_name, model_info in models_info.items():
			model_class = model_info["model_class"]
			if model_class in categorized_models:
				categorized_models[model_class].append(model_name)

		for model_class_item in [model_class_item1, model_class_item2, model_class_item3, model_class_item4]:
			self.tree.addTopLevelItem(model_class_item)
			model_class_item.setCheckState(0, Qt.Unchecked)
			model_class_item.setData(0, Qt.ItemDataRole.AccessibleTextRole, model_class_item.text(0))

			models = categorized_models[model_class_item.text(0)]

			for model_name in models:
				item = QTreeWidgetItem()
				item.setText(0, model_name)
				item.setCheckState(0, Qt.Unchecked)
				item.setData(0, Qt.ItemDataRole.AccessibleTextRole, model_name)
				model_class_item.addChild(item)

	def treeItemChanged(self, item):
		parent_text = item.parent().text(0) if item.parent() else None

		if item.checkState(0) == Qt.Checked:
			if item.childCount() > 0:
				for i in range(item.childCount()):
					item.child(i).setCheckState(0, Qt.Checked)
			else:
				self.addModelToDownload(parent_text, item.text(0))
		elif item.checkState(0) == Qt.Unchecked:
			if item.childCount() > 0:
				for i in range(item.childCount()):
					item.child(i).setCheckState(0, Qt.Unchecked)
			else:
				self.removeModelFromDownload(parent_text, item.text(0))

		self.updateParentItem(item)

	def addModelToDownload(self, category, model_name):
		if category in self.model_to_download and model_name not in self.model_to_download[category]:
			self.model_to_download[category].append(model_name)
			self.addButtonToLayout(model_name)

	def removeModelFromDownload(self, category, model_name):
		if category in self.model_to_download and model_name in self.model_to_download[category]:
			self.model_to_download[category].remove(model_name)
			self.removeButtonFromLayout(model_name)

	def updateParentItem(self, item):
		parent = item.parent()
		if parent is None:
			return

		selectedCount = 0
		childCount = parent.childCount()

		for i in range(childCount):
			childItem = parent.child(i)
			if childItem.checkState(0) == Qt.Checked:
				selectedCount += 1

		if selectedCount == 0:
			parent.setCheckState(0, Qt.Unchecked)
		elif 0 < selectedCount < childCount:
			parent.setCheckState(0, Qt.PartiallyChecked)
		else:
			parent.setCheckState(0, Qt.Checked)

	def addButtonToLayout(self, model_name):
		tag_widget = TagWidget(model_name, self.tr("Remove from download list"))
		tag_widget.deleteSignal.connect(self.removeTagFromLayout)
		self.flow_layout.addWidget(tag_widget)

	def removeButtonFromLayout(self, model_name):
		for i in range(self.flow_layout.count()):
			widget = self.flow_layout.itemAt(i).widget()
			if isinstance(widget, TagWidget) and widget.text == model_name:
				self.flow_layout.removeWidget(widget)
				widget.deleteLater()
				break
		self.flow_layout.update()

	def removeTagFromLayout(self, model_name):
		for item in [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]:
			for child_item in [item.child(i) for i in range(item.childCount())]:
				if child_item.text(0) == model_name:
					child_item.setCheckState(0, Qt.Unchecked)

	def generate_urls(self):
		self.model_urls.clear()
		self.total_files = 0

		for category, models in self.model_to_download.items():
			for model in models:
				url = hf_hub_url(repo_id="Sucial/MSST-WebUI", filename=f"All_Models/{category}/{model}", endpoint=HF_ENDPOINT)
				self.model_urls.append(url)
				self.total_files += 1

	def download_all_models(self):
		self.generate_urls()
		if not self.model_urls:
			accessible_info_bar(
				"error",
				title="ERROR",
				content=self.tr("Please select models to download first!"),
				isClosable=True,
				position=InfoBarPosition.TOP,
				duration=5000,
				parent=self,
				close_text=self.tr("Close"),
			)
			return
		accessible_info_bar(
			"info",
			title="INFO",
			content=self.tr("Downloading models, Please do not click repeatedly..."),
			isClosable=True,
			position=InfoBarPosition.TOP,
			duration=5000,
			parent=self,
			close_text=self.tr("Close"),
		)
		self.download_action.setEnabled(False)
		self.aria2_action.setEnabled(False)
		self.total_progress_bar.setValue(0)
		self.single_progress_bar.setValue(0)
		preparing_text = self.tr("Preparing to download...")
		self.download_speed_label.setText(preparing_text)
		self.download_speed_label.setAccessibleName(preparing_text)
		target_dir = "./pretrain"
		self.total_progress = 0

		# 启动下载线程
		self.download_thread = DownloadThread(self.model_urls, target_dir)
		self.download_thread.update_single_progress.connect(self.single_progress_bar.setValue)
		self.download_thread.update_total_progress.connect(self.total_progress_bar.setValue)
		self.download_thread.update_speed.connect(self.update_download_speed)
		self.download_thread.completed.connect(self.download_finished)
		self.download_thread.failed.connect(self.download_failed)
		self.download_thread.start()

	def update_download_speed(self, speed):
		# print(f"当前下载速度: {speed}")
		text = self.tr("Download speed: {speed}").format(speed=speed)
		self.download_speed_label.setText(text)
		self.download_speed_label.setAccessibleName(text)

	def download_finished(self):
		completed_text = self.tr("Download task completed!")
		self.download_speed_label.setText(completed_text)
		self.download_speed_label.setAccessibleName(completed_text)
		self.download_action.setEnabled(True)
		self.aria2_action.setEnabled(True)
		accessible_info_bar(
			"success",
			title="SUCCESS",
			content=completed_text,
			isClosable=True,
			position=InfoBarPosition.TOP,
			duration=5000,
			parent=self,
			close_text=self.tr("Close"),
		)
		self.clearModels()

	def download_failed(self, error):
		failed_text = self.tr("Download failed: {error}").format(error=error)
		self.download_speed_label.setText(failed_text)
		self.download_speed_label.setAccessibleName(failed_text)
		self.download_action.setEnabled(True)
		self.aria2_action.setEnabled(True)
		accessible_info_bar(
			"error",
			title="ERROR",
			content=failed_text,
			isClosable=True,
			position=InfoBarPosition.TOP,
			duration=-1,
			parent=self,
			close_text=self.tr("Close"),
		)

	def send_to_aria2(self):
		self.generate_urls()
		if not self.model_urls:
			accessible_info_bar(
				"error",
				title="ERROR",
				content=self.tr("Please select models to download first!"),
				isClosable=True,
				position=InfoBarPosition.TOP,
				duration=5000,
				parent=self,
				close_text=self.tr("Close"),
			)
			return

		download_dir = os.path.join(os.getcwd(), "pretrain")
		flag = True

		for url in self.model_urls:
			model_filename = url.split("/")[-1]
			category = url.split("/")[-2]

			target_path = f"{download_dir}/{category}"
			json_rpc_data = {"jsonrpc": "2.0", "method": "aria2.addUri", "id": 1, "params": [f"token:{ARIA2_RPC_SECRET}", [url], {"dir": target_path, "out": model_filename}]}

			try:
				response = requests.post(ARIA2_RPC_URL, data=json.dumps(json_rpc_data))
			except requests.exceptions.ConnectionError as e:
				accessible_info_bar(
					"error",
					title="ERROR",
					content=self.tr("Failed to connect to Aria2 RPC: {e}. Please check if the Aria2 service is running.").format(e=e),
					isClosable=True,
					position=InfoBarPosition.TOP,
					duration=-1,
					parent=self,
					close_text=self.tr("Close"),
				)
				flag = False
				continue

			if response.status_code != 200:
				accessible_info_bar(
					"error",
					title="ERROR",
					content=self.tr("Failed to submit the task, error message: {response.text}").format(response=response),
					isClosable=True,
					position=InfoBarPosition.TOP,
					duration=5000,
					parent=self,
					close_text=self.tr("Close"),
				)
				flag = False

		if flag:
			accessible_info_bar(
				"success",
				title="SUCCESS",
				content=self.tr("The download task(s) has been submitted to Aria2."),
				isClosable=True,
				position=InfoBarPosition.TOP,
				duration=5000,
				parent=self,
				close_text=self.tr("Close"),
			)
			self.clearModels()

	def clearModels(self):
		for i in range(self.tree.topLevelItemCount()):
			item = self.tree.topLevelItem(i)
			item.setCheckState(0, Qt.Unchecked)

			for j in range(item.childCount()):
				child = item.child(j)
				child.setCheckState(0, Qt.Unchecked)

		# print(self.model_to_download)
		# print(self.model_urls)
