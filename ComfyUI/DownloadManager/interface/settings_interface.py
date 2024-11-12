from PySide6.QtWidgets import QWidget, QSpacerItem, QFrame, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from qfluentwidgets import (setTheme, ScrollArea, setThemeColor, ExpandLayout, SettingCardGroup, OptionsSettingCard, 
                            CustomColorSettingCard, SettingCard, InfoBar, LineEdit, TitleLabel
                            )
from qfluentwidgets import FluentIcon as FIF
from common.config import cfg


class settingsInterface(QFrame):
    def __init__(self, parent = None):
        super().__init__(parent)
        
        self.setObjectName('settingsInterface')
        self.setupUI()
        
    def setupUI(self):
        self.layout = QVBoxLayout(self)
        
        self.settingLabel = TitleLabel("设置", self)
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
        
        self.personalGroup.addSettingCards([self.themeCard, self.themeColorCard])
        self.settingGroup.addSettingCards([self.aria2Card, self.hfEndpointCard])

        self.cardsLayout.addWidget(self.personalGroup)
        self.cardsLayout.addWidget(self.settingGroup)

        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)
        
    
    def initSettingCards(self):    
        self.personalGroup = SettingCardGroup(
            "个性化", self.widget)
        self.themeCard = OptionsSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            "应用主题",
            "选择应用的主题",
            texts=[
                self.tr('Light'), self.tr('Dark'),
                self.tr('Use system setting')
            ],
            parent=self.personalGroup
        )
        self.themeColorCard = CustomColorSettingCard(
            cfg.themeColor,
            FIF.PALETTE,
            "主题颜色",
            "选择应用的主题颜色",
            self.personalGroup
        )

        self.settingGroup = SettingCardGroup(
            "配置", self.widget)
        self.aria2Card = SettingCard(
            FIF.GLOBE,
            "Aria2 RPC",
            "配置Aria2 RPC的地址和端口",
            self.settingGroup
        )
        aria2_port_line_edit = LineEdit()
        aria2_port_line_edit.setText(str(cfg.get(cfg.aria2_port)))
        int_validator = QIntValidator(0, 100000)
        aria2_port_line_edit.setValidator(int_validator)
        self.aria2Card.hBoxLayout.addWidget(aria2_port_line_edit)
        self.aria2Card.hBoxLayout.addSpacerItem(QSpacerItem(20, 20))
        aria2_port_line_edit.textChanged.connect(self.setAria2Port)

        self.hfEndpointCard = SettingCard(
            FIF.APPLICATION,
            "Hugging Face Endpoint",
            "设置Hugging Face下载源地址",
            self.settingGroup
        )

        hf_endpoint_line_edit = LineEdit()
        hf_endpoint_line_edit.setText(str(cfg.get(cfg.hf_endpoint)))
        hf_endpoint_line_edit.setFixedWidth(len(hf_endpoint_line_edit.text()) * 8)
        self.hfEndpointCard.hBoxLayout.addWidget(hf_endpoint_line_edit)
        self.hfEndpointCard.hBoxLayout.addSpacerItem(QSpacerItem(20, 20))
        hf_endpoint_line_edit.textChanged.connect(self.setHfEndpoint)

        self.connectSignalToSlot()    

    def setHfEndpoint(self, endpoint):
        cfg.set(cfg.hf_endpoint, endpoint)

    def setAria2Port(self, port):
        port = int(port)
        cfg.set(cfg.aria2_port, port)

    def connectSignalToSlot(self):
        cfg.appRestartSig.connect(self.showRestartTooltip)

        # personalization
        cfg.themeChanged.connect(setTheme)
        self.themeColorCard.colorChanged.connect(lambda c: setThemeColor(c))
        
    def showRestartTooltip(self):
        InfoBar.success(
            'Updated successfully',
            'Configuration takes effect after restart',
            duration=1500,
            parent=self
        )