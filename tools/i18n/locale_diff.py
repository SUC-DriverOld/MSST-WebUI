import json
import os
import re
import opencc


def extract_i18n_strings(file_content):
	pattern = re.compile(r'i18n\("([^"]+)"\)')
	return pattern.findall(file_content)


def process_py_file(file_path):
	i18n_dict = {}

	with open(file_path, "r", encoding="utf-8") as f:
		content = f.read()
		i18n_strings = extract_i18n_strings(content)
		for string in i18n_strings:
			i18n_dict[string] = "text"
	return i18n_dict


def save_to_json(data, output_file):
	with open(output_file, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=4)


def scan_files(input_paths):
	i18n_dict = {}
	for path in input_paths:
		if os.path.isfile(path):
			file_dict = process_py_file(path)
			i18n_dict.update(file_dict)
		elif os.path.isdir(path):
			for root, _, files in os.walk(path):
				for file in files:
					if file.endswith(".py"):
						file_path = os.path.join(root, file)
						file_dict = process_py_file(file_path)
						i18n_dict.update(file_dict)
	return i18n_dict


def locale_diff(template, target):
	with open(template, "r", encoding="utf-8") as f:
		template_key = json.load(f)
	with open(target, "r", encoding="utf-8") as f:
		target_key = json.load(f)
	for key in template_key:
		if key in target_key:
			template_key[key] = target_key[key]
		else:
			print("Missing: " + key)
	with open(target, "w", encoding="utf-8") as f:
		json.dump(template_key, f, ensure_ascii=False, indent=4)


def sort(target):
	old = {}
	new = {}

	with open(target, "r", encoding="utf-8") as f:
		data = json.load(f)
	for key in data:
		if data[key] == "text":
			new[key] = data[key]
		else:
			old[key] = data[key]
	data = {**old, **new}
	with open(target, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=4)


def opencc_convert():
	cc = opencc.OpenCC("s2t")

	with open("locale/template.json", "r", encoding="utf-8") as f:
		template = json.load(f)
		for key in template:
			template[key] = cc.convert(key)

	with open("locale/zh_TW.json", "w", encoding="utf-8") as f:
		json.dump(template, f, ensure_ascii=False, indent=4)
	with open("locale/zh_HK.json", "w", encoding="utf-8") as f:
		json.dump(template, f, ensure_ascii=False, indent=4)
	with open("locale/zh_SG.json", "w", encoding="utf-8") as f:
		json.dump(template, f, ensure_ascii=False, indent=4)


def main():
	input_paths = ["E:\\vs\\MSST-WebUI\\webui", "E:\\vs\\MSST-WebUI\\webUI.py", "E:\\vs\\MSST-WebUI\\tools\\webUI_for_clouds\\webUI_for_clouds.py"]
	target = ["locale/en_US.json", "locale/ja_JP.json", "locale/emoji.json", "locale/ko_KR.json"]
	template = "locale/template.json"

	# step 1: scan files and extract i18n strings
	i18n_dict = scan_files(input_paths)
	save_to_json(i18n_dict, template)

	# step 2: diff locale files
	for t in target:
		locale_diff(template, t)
		sort(t)

	# step 3: opencc convert
	opencc_convert()


if __name__ == "__main__":
	main()
	print("Done")
