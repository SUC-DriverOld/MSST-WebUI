import json
import locale
import os


def load_language_list(language):
	with open(f"tools/i18n/locale/{language}.json", "r", encoding="utf-8") as f:
		language_list = json.load(f)
	return language_list


class I18nAuto:
	def __init__(self, language):
		if language == "zh_CN":
			self.language = language
			return
		if language == "Auto":
			language = locale.getdefaultlocale()[0]
		if not os.path.exists(f"tools/i18n/locale/{language}.json"):
			language = "en_US"
		self.language = language
		self.language_map = load_language_list(language)

	def __call__(self, key):
		if self.language == "zh_CN":
			return key
		return self.language_map.get(key, key)
