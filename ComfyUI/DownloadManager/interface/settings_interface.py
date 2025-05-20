from PySide6.QtWidgets import QWidget, QSpacerItem, QFrame, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from qfluentwidgets import (
	setTheme,
	ScrollArea,
	setThemeColor,
	SettingCardGroup,
	OptionsSettingCard,
	PasswordLineEdit,
	CustomColorSettingCard,
	SettingCard,
	InfoBar,
	LineEdit,
	TitleLabel,
	ComboBoxSettingCard,
)
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.DownloadManager.common.config import cfg


class SettingsInterface(QFrame):
	def __init__(self, parent=None):
		super().__init__(parent)

		self.setObjectName("SettingsInterface")
		self.setupUI()

	def setupUI(self):
		self.layout = QVBoxLayout(self)

		self.settingLabel = TitleLabel(self.tr("Settings"), self)
		self.layout.addWidget(self.settingLabel)

		self.scroll_area = ScrollArea(self)
		self.scroll_area.setStyleSheet("background-color: transparent;")
		self.scroll_area.setWidgetResizable(True)

		self.widget = QWidget(self)
		self.cardsLayout = QVBoxLayout(self.widget)

		self.scroll_area.setWidget(self.widget)

		self.initSettingCards()
		self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		# self.scroll_area.setViewportMargins(0, 80, 0, 20)

		self.personalGroup.addSettingCards([self.themeCard, self.themeColorCard, self.languageCard])
		self.settingGroup.addSettingCards([self.aria2Card, self.aria2SecretCard, self.hfEndpointCard])

		self.cardsLayout.addWidget(self.personalGroup)
		self.cardsLayout.addWidget(self.settingGroup)

		self.layout.addWidget(self.scroll_area)
		self.setLayout(self.layout)

	def initSettingCards(self):
		self.personalGroup = SettingCardGroup(self.tr("Personalization"), self.widget)
		self.themeCard = OptionsSettingCard(
			cfg.themeMode,
			FIF.BRUSH,
			self.tr("Theme"),
			self.tr("Choose the theme of the application"),
			texts=[self.tr("Light"), self.tr("Dark"), self.tr("Use system setting")],
			parent=self.personalGroup,
		)
		self.themeColorCard = CustomColorSettingCard(cfg.themeColor, FIF.PALETTE, self.tr("Theme Color"), self.tr("Choose the color of the theme"), self.personalGroup)

		self.languageCard = ComboBoxSettingCard(
			cfg.language,
			FIF.LANGUAGE,
			self.tr("Language"),
			self.tr("Set your preferred language for UI"),
			texts=["简体中文", "日本語", "English", self.tr("Use system setting")],
			parent=self.personalGroup,
		)

		self.settingGroup = SettingCardGroup(self.tr("Configuration"), self.widget)
		self.aria2Card = SettingCard(FIF.GLOBE, "Aria2 RPC", self.tr("Set the port of Aria2 RPC server"), self.settingGroup)
		aria2_port_line_edit = LineEdit()
		aria2_port_line_edit.setText(str(cfg.get(cfg.aria2_port)))
		int_validator = QIntValidator(0, 100000)
		aria2_port_line_edit.setValidator(int_validator)
		self.aria2Card.hBoxLayout.addWidget(aria2_port_line_edit)
		self.aria2Card.hBoxLayout.addSpacerItem(QSpacerItem(20, 20))
		aria2_port_line_edit.textChanged.connect(self.setAria2Port)

		self.aria2SecretCard = SettingCard(FIF.SETTING, "Aria2 Secret", self.tr("Set the secret token of Aria2 RPC server"), self.settingGroup)
		aria2_secret_line_edit = PasswordLineEdit()
		aria2_secret_line_edit.setClearButtonEnabled(True)
		aria2_secret_line_edit.setPlaceholderText("Secret key")
		aria2_secret_line_edit.setText(str(cfg.get(cfg.aria2_secret)))
		self.aria2SecretCard.hBoxLayout.addWidget(aria2_secret_line_edit)
		self.aria2SecretCard.hBoxLayout.addSpacerItem(QSpacerItem(20, 20))
		aria2_secret_line_edit.textChanged.connect(self.setAria2Secret)

		self.hfEndpointCard = SettingCard(FIF.APPLICATION, "Hugging Face Endpoint", self.tr("Set up HuggingFace (mirror) site."), self.settingGroup)

		hf_endpoint_line_edit = LineEdit()
		hf_endpoint_line_edit.setText(str(cfg.get(cfg.hf_endpoint)))
		hf_endpoint_line_edit.setFixedWidth(len(hf_endpoint_line_edit.text()) * 8)
		self.hfEndpointCard.hBoxLayout.addWidget(hf_endpoint_line_edit)
		self.hfEndpointCard.hBoxLayout.addSpacerItem(QSpacerItem(20, 20))
		# hf_endpoint_line_edit.textChanged.connect(self.setHfEndpoint)
		hf_endpoint_line_edit.editingFinished.connect(lambda: self.setHfEndpoint(hf_endpoint_line_edit.text()))

		self.connectSignalToSlot()

	def setHfEndpoint(self, endpoint):
		cfg.set(cfg.hf_endpoint, endpoint)

	def setAria2Port(self, port):
		port = int(port)
		cfg.set(cfg.aria2_port, port)

	def setAria2Secret(self, secret):
		cfg.set(cfg.aria2_secret, secret)

	def connectSignalToSlot(self):
		cfg.appRestartSig.connect(self.showRestartTooltip)

		# personalization
		cfg.themeChanged.connect(setTheme)
		self.themeColorCard.colorChanged.connect(lambda c: setThemeColor(c))

	def showRestartTooltip(self):
		InfoBar.success(self.tr("Settings saved"), self.tr("Please restart the application to apply the changes"), duration=1500, parent=self)
