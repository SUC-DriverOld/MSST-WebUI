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
	LineEdit,
	TitleLabel,
	ComboBoxSettingCard,
)
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.DownloadManager.common.accessibility import accessible_info_bar, make_keyboard_activatable, set_accessible
from ComfyUI.DownloadManager.common.config import cfg


class SettingsInterface(QFrame):
	def __init__(self, parent=None):
		super().__init__(parent)

		self.setObjectName("SettingsInterface")
		self.setupUI()

	def setupUI(self):
		self.layout = QVBoxLayout(self)

		self.settingLabel = TitleLabel(self.tr("Settings"), self)
		set_accessible(self.settingLabel, self.tr("Settings"))
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
		self.configureExpandCard(self.themeCard, self.tr("Theme"), self.tr("Choose the theme of the application"))
		self.themeColorCard = CustomColorSettingCard(cfg.themeColor, FIF.PALETTE, self.tr("Theme Color"), self.tr("Choose the color of the theme"), self.personalGroup)
		self.configureExpandCard(self.themeColorCard, self.tr("Theme Color"), self.tr("Choose the color of the theme"))
		set_accessible(
			self.themeColorCard.chooseColorButton,
			self.tr("Theme Color"),
			self.tr("Choose the color of the theme"),
			tooltip=self.tr("Choose the color of the theme"),
			focusable=True,
		)

		self.languageCard = ComboBoxSettingCard(
			cfg.language,
			FIF.LANGUAGE,
			self.tr("Language"),
			self.tr("Set your preferred language for UI"),
			texts=["简体中文", "日本語", "English", self.tr("Use system setting")],
			parent=self.personalGroup,
		)
		set_accessible(
			self.languageCard.comboBox,
			self.tr("Language"),
			self.tr("Set your preferred language for UI"),
			tooltip=self.tr("Set your preferred language for UI"),
			focusable=True,
		)
		make_keyboard_activatable(self.languageCard.comboBox, self.languageCard.comboBox._toggleComboMenu, allow_alt_down=True)

		self.settingGroup = SettingCardGroup(self.tr("Configuration"), self.widget)
		self.aria2Card = SettingCard(FIF.GLOBE, "Aria2 RPC", self.tr("Set the port of Aria2 RPC server"), self.settingGroup)
		self.aria2_port_line_edit = LineEdit()
		self.aria2_port_line_edit.setText(str(cfg.get(cfg.aria2_port)))
		self.aria2_port_line_edit.setValidator(QIntValidator(1, 65535))
		set_accessible(
			self.aria2_port_line_edit,
			"Aria2 RPC",
			self.tr("Set the port of Aria2 RPC server"),
			tooltip=self.tr("Set the port of Aria2 RPC server"),
			focusable=True,
		)
		self.aria2Card.hBoxLayout.addWidget(self.aria2_port_line_edit)
		self.aria2Card.hBoxLayout.addSpacerItem(QSpacerItem(20, 20))
		self.aria2_port_line_edit.textChanged.connect(self.setAria2Port)

		self.aria2SecretCard = SettingCard(FIF.SETTING, "Aria2 Secret", self.tr("Set the secret token of Aria2 RPC server"), self.settingGroup)
		self.aria2_secret_line_edit = PasswordLineEdit()
		self.aria2_secret_line_edit.setClearButtonEnabled(True)
		self.aria2_secret_line_edit.setPlaceholderText("Secret key")
		self.aria2_secret_line_edit.setText(str(cfg.get(cfg.aria2_secret)))
		set_accessible(
			self.aria2_secret_line_edit,
			"Aria2 Secret",
			self.tr("Set the secret token of Aria2 RPC server"),
			tooltip=self.tr("Set the secret token of Aria2 RPC server"),
			focusable=True,
		)
		clear_secret_text = self.tr("Clear Aria2 Secret")
		set_accessible(self.aria2_secret_line_edit.clearButton, clear_secret_text, tooltip=clear_secret_text, focusable=True)
		view_secret_text = self.tr("Show or hide Aria2 Secret")
		self.aria2_secret_line_edit.viewButton.removeEventFilter(self.aria2_secret_line_edit)
		self.aria2_secret_line_edit.viewButton.setCheckable(True)
		self.aria2_secret_line_edit.viewButton.toggled.connect(self.aria2_secret_line_edit.setPasswordVisible)
		set_accessible(self.aria2_secret_line_edit.viewButton, view_secret_text, tooltip=view_secret_text, focusable=True)
		self.aria2SecretCard.hBoxLayout.addWidget(self.aria2_secret_line_edit)
		self.aria2SecretCard.hBoxLayout.addSpacerItem(QSpacerItem(20, 20))
		self.aria2_secret_line_edit.textChanged.connect(self.setAria2Secret)

		self.hfEndpointCard = SettingCard(FIF.APPLICATION, "Hugging Face Endpoint", self.tr("Set up HuggingFace (mirror) site."), self.settingGroup)

		self.hf_endpoint_line_edit = LineEdit()
		self.hf_endpoint_line_edit.setText(str(cfg.get(cfg.hf_endpoint)))
		self.hf_endpoint_line_edit.setMinimumWidth(260)
		set_accessible(
			self.hf_endpoint_line_edit,
			"Hugging Face Endpoint",
			self.tr("Set up HuggingFace (mirror) site."),
			tooltip=self.tr("Set up HuggingFace (mirror) site."),
			focusable=True,
		)
		self.hfEndpointCard.hBoxLayout.addWidget(self.hf_endpoint_line_edit)
		self.hfEndpointCard.hBoxLayout.addSpacerItem(QSpacerItem(20, 20))
		# hf_endpoint_line_edit.textChanged.connect(self.setHfEndpoint)
		self.hf_endpoint_line_edit.editingFinished.connect(lambda: self.setHfEndpoint(self.hf_endpoint_line_edit.text()))

		self.connectSignalToSlot()

	def configureExpandCard(self, card, name, description):
		set_accessible(card, name, description)
		card.card.expandButton.setCheckable(True)
		set_accessible(card.card.expandButton, name, description, tooltip=description, focusable=True)

	def setHfEndpoint(self, endpoint):
		cfg.set(cfg.hf_endpoint, endpoint)

	def setAria2Port(self, port):
		if not port:
			return
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
		accessible_info_bar(
			"success",
			title=self.tr("Settings saved"),
			content=self.tr("Please restart the application to apply the changes"),
			duration=1500,
			parent=self,
			close_text=self.tr("Close"),
		)
