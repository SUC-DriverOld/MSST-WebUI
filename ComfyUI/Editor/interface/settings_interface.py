from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout
from PySide6.QtCore import Qt
from qfluentwidgets import (setTheme, ScrollArea, setThemeColor, SettingCardGroup, OptionsSettingCard,
                            CustomColorSettingCard, InfoBar, TitleLabel, ComboBoxSettingCard, SettingCard
                            )
from qfluentwidgets import FluentIcon as FIF
from ComfyUI.Editor.common.config import cfg


class SettingsInterface(QFrame):
    def __init__(self, parent = None):
        super().__init__(parent)
        
        self.setObjectName('SettingsInterface')
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

        self.cardsLayout.addWidget(self.personalGroup)

        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)
        
    
    def initSettingCards(self):    
        self.personalGroup = SettingCardGroup(
            self.tr("Personalization"), self.widget)
        self.themeCard = OptionsSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            self.tr("Theme"),
            self.tr("Choose the theme of the application"),
            texts=[
                self.tr('Light'), self.tr('Dark'),
                self.tr('Use system setting')
            ],
            parent=self.personalGroup
        )
        self.themeColorCard = CustomColorSettingCard(
            cfg.themeColor,
            FIF.PALETTE,
            self.tr("Theme Color"),
            self.tr("Choose the color of the theme"),
            self.personalGroup
        )

        self.languageCard = ComboBoxSettingCard(
            cfg.language,
            FIF.LANGUAGE,
            self.tr('Language'),
            self.tr('Set your preferred language for UI'),
            texts=['简体中文', '日本語', 'English', self.tr('Use system setting')],
            parent=self.personalGroup
        )
        
        self.settingsGroup = SettingCardGroup(
            self.tr("Settings"), self.widget)

        self.connectSignalToSlot()    

    def connectSignalToSlot(self):
        cfg.appRestartSig.connect(self.showRestartTooltip)

        # personalization
        cfg.themeChanged.connect(setTheme)
        self.themeColorCard.colorChanged.connect(lambda c: setThemeColor(c))
        
    def showRestartTooltip(self):
        InfoBar.success(
            self.tr("Settings saved"),
            self.tr("Please restart the application to apply the changes"),
            duration=1500,
            parent=self
        )