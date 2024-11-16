from qfluentwidgets import FluentWindow, NavigationItemPosition
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.DownloadManager.interface.download_interface import downloadInterface
from ComfyUI.DownloadManager.interface.manager_interface import managerInterface
from ComfyUI.DownloadManager.interface.settings_interface import settingsInterface


class DownloadManager(FluentWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        download_interface = downloadInterface(self)
        manager_interface = managerInterface(self)
        settings_interface = settingsInterface(self)
        self.addSubInterface(download_interface, FIF.DOWNLOAD, "Download Center")
        self.addSubInterface(manager_interface, FIF.TAG, "Manager")
        self.addSubInterface(settings_interface, FIF.SETTING, "Settings", NavigationItemPosition.BOTTOM)
        self.resize(900, 600)