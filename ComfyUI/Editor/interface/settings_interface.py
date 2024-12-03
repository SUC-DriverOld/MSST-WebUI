from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QFileDialog
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import Qt, QUrl
from qfluentwidgets import (setTheme, ScrollArea, setThemeColor, SettingCardGroup, OptionsSettingCard,
                            CustomColorSettingCard, InfoBar, TitleLabel, ComboBoxSettingCard, SettingCard,
                            PushSettingCard
                            )
from qfluentwidgets import FluentIcon as FIF
from pathlib import Path
from ComfyUI.Editor.common.config import cfg


class SettingsInterface(QFrame):
    def __init__(self, parent = None):
        super().__init__(parent)
        self._update_enabled = False
        
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
        self.settingsGroup.addSettingCards([self.presetPathCard, self.tmpPathCard, self.logPathCard])
        self.cardsLayout.addWidget(self.settingsGroup)

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
        self.presetPathCard = PushSettingCard(
            icon=FIF.FOLDER,
            text=self.tr("Select Preset Folder"),
            content=cfg.get(cfg.preset_path),
            title=self.tr("Preset Path"),
        )
        self.tmpPathCard = PushSettingCard(
            icon=FIF.FOLDER,
            text=self.tr("Select Temp Folder"),
            content=cfg.get(cfg.tmp_path),
            title=self.tr("Temp Path"),
        )
        self.logPathCard = PushSettingCard(
            icon=FIF.FOLDER,
            text=self.tr("Select Log Folder"),
            content=cfg.get(cfg.log_path),
            title=self.tr("Log Path"),
        )

        self.connectSignalToSlot()    

    def connectSignalToSlot(self):
        cfg.appRestartSig.connect(self.showRestartTooltip)

        # personalization
        cfg.themeChanged.connect(setTheme)
        self.themeColorCard.colorChanged.connect(lambda c: setThemeColor(c))
        
        # editor settings
        self.presetPathCard.clicked.connect(self.selectPresetPath)
        self.tmpPathCard.clicked.connect(self.selectTmpPath)
        self.logPathCard.clicked.connect(self.selectLogPath)
        
    def showRestartTooltip(self):
        InfoBar.success(
            self.tr("Settings saved"),
            self.tr("Please restart the application to apply the changes"),
            duration=1500,
            parent=self
        )
        
    def setUpdatesEnabled(self, enabled):
        self._update_enabled = enabled
        super().setUpdatesEnabled(enabled)    
        
    def selectPresetPath(self):
        """Opens a directory selection dialog and updates the preset path configuration"""
        
        # Get the current preset path or default to home directory
        current_path = Path(cfg.get(cfg.preset_path))
        
        # Open directory selection dialog
        new_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Preset Folder"),
            str(current_path),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        # Update configuration if a new path was selected
        if new_path:
            # Convert to Path object to ensure proper path handling
            path = Path(new_path)
            
            # Update the configuration
            cfg.set(cfg.preset_path, str(path))
            
            # Update the card content to show the new path
            self.presetPathCard.setContent(str(path))
            
            # Show success message
            InfoBar.success(
                self.tr("Path Updated"),
                self.tr("Preset folder path has been updated successfully"),
                duration=1500,
                parent=self
            )
            
    def selectTmpPath(self):        
        current_path = Path(cfg.get(cfg.tmp_path))
        
        new_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Temp Folder"),
            str(current_path),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if new_path:
            path = Path(new_path)
            
            cfg.set(cfg.tmp_path, str(path))
            
            self.tmpPathCard.setContent(str(path))
            
            InfoBar.success(
                self.tr("Path Updated"),
                self.tr("Temp folder path has been updated successfully"),
                duration=1500,
                parent=self
            )    
            
    def selectLogPath(self):
        current_path = Path(cfg.get(cfg.log_path))
        
        new_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Log Folder"),
            str(current_path),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if new_path:
            path = Path(new_path)
            
            cfg.set(cfg.log_path, str(path))
            
            self.logPathCard.setContent(str(path))
            
            InfoBar.success(
                self.tr("Path Updated"),
                self.tr("Log folder path has been updated successfully"),
                duration=1500,
                parent=self
            )            