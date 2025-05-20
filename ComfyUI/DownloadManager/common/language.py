from enum import Enum
from PySide6.QtCore import QLocale
from qfluentwidgets import ConfigSerializer


class Language(Enum):
	CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.China)
	# CHINESE_TRADITIONAL = QLocale(QLocale.Chinese, QLocale.HongKong)
	JAPANESE = QLocale(QLocale.Japanese)
	ENGLISH = QLocale(QLocale.English)
	AUTO = QLocale()


class LanguageSerializer(ConfigSerializer):
	def serialize(self, language):
		return language.value.name() if language != Language.AUTO else "Auto"

	def deserialize(self, value: str):
		return Language(QLocale(value)) if value != "Auto" else Language.AUTO
