import re
import json

def extract_i18n_strings(file_content):
    pattern = re.compile(r'i18n\("([^"]+)"\)')
    return pattern.findall(file_content)

def process_py_file(file_path):
    i18n_dict = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        i18n_strings = extract_i18n_strings(content)
        for i, string in enumerate(i18n_strings, 1):
            i18n_dict[string] = "text"

    return i18n_dict

def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

file_path = "E:\\vs\\MSST-WebUI\\webUI.py"
output_file = "E:\\vs\\MSST-WebUI\\tools\\i18n\\locale\\template.json"
i18n_dict = process_py_file(file_path)
save_to_json(i18n_dict, output_file)
