from PySide6.QtCore import QEvent
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import FluentWindow, NavigationItemPosition

from ComfyUI.DownloadManager.common.accessibility import announce, make_keyboard_activatable, set_accessible
from ComfyUI.DownloadManager.interface.download_interface import DownloadInterface
from ComfyUI.DownloadManager.interface.manager_interface import ManagerInterface
from ComfyUI.DownloadManager.interface.settings_interface import SettingsInterface


class DownloadManager(FluentWindow):
	__AUTHOR__ = "MSST-WebUI-Develop-Team KitsuneX07"
	__VERSION__ = "1.1.0"

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setupUI()

	def setupUI(self):
		self.download_interface = DownloadInterface(self)
		self.manager_interface = ManagerInterface(self)
		self.settings_interface = SettingsInterface(self)

		download_text = self.download_interface.tr("Download Center")
		manager_text = self.manager_interface.tr("Local Model Library")
		settings_text = self.settings_interface.tr("Settings")
		self.download_navigation_item = self.addSubInterface(self.download_interface, FIF.DOWNLOAD, download_text)
		self.manager_navigation_item = self.addSubInterface(self.manager_interface, FIF.TAG, manager_text)
		self.settings_navigation_item = self.addSubInterface(self.settings_interface, FIF.SETTING, settings_text, NavigationItemPosition.BOTTOM)

		for item, name in [
			(self.download_navigation_item, download_text),
			(self.manager_navigation_item, manager_text),
			(self.settings_navigation_item, settings_text),
		]:
			set_accessible(item, name, tooltip=name, focusable=True)
			make_keyboard_activatable(item, item.click)
			item.selectedChanged.connect(lambda selected, name=name: announce(self, name) if selected else None)

		panel = self.navigationInterface.panel
		set_accessible(panel.menuButton, panel.menuButton.toolTip(), tooltip=panel.menuButton.toolTip(), focusable=True)
		set_accessible(panel.returnButton, panel.returnButton.toolTip(), tooltip=panel.returnButton.toolTip(), focusable=True)
		self.update_title_bar_accessibility()
		self.setWindowTitle("Download Manager v" + self.__VERSION__ + ", developed by " + self.__AUTHOR__)
		set_accessible(self, self.windowTitle())
		self.resize(900, 600)

	def update_title_bar_accessibility(self):
		set_accessible(self.titleBar.minBtn, self.tr("Minimize"), tooltip=self.tr("Minimize"), focusable=True)
		maximize_text = self.tr("Restore") if self.isMaximized() else self.tr("Maximize")
		set_accessible(self.titleBar.maxBtn, maximize_text, tooltip=maximize_text, focusable=True)
		set_accessible(self.titleBar.closeBtn, self.tr("Close"), tooltip=self.tr("Close"), focusable=True)

	def changeEvent(self, event):  # noqa: N802 - Qt virtual method name
		super().changeEvent(event)
		if event.type() == QEvent.Type.WindowStateChange:
			self.update_title_bar_accessibility()
