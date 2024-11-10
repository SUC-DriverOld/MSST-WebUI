import os
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
        for string in i18n_strings:
            i18n_dict[string] = "text"

    return i18n_dict

def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def scan_files(input_paths):
    i18n_dict = {}

    for path in input_paths:
        if os.path.isfile(path):
            print(f"Processing file: {path}")
            file_dict = process_py_file(path)
            i18n_dict.update(file_dict)
        elif os.path.isdir(path):
            print(f"Scanning directory: {path}")
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        print(f"Processing file: {file_path}")
                        file_dict = process_py_file(file_path)
                        i18n_dict.update(file_dict)

    return i18n_dict

input_paths = [
    "E:\\vs\\MSST-WebUI\\webui",
    "E:\\vs\\MSST-WebUI\\webUI.py",
    "E:\\vs\\MSST-WebUI\\app.py",
    "E:\\vs\\MSST-WebUI\\tools\\webUI_for_clouds\\webUI_for_clouds.py"
]
output_file = "E:\\vs\\MSST-WebUI\\tools\\i18n\\locale\\template.json"
i18n_dict = scan_files(input_paths)
save_to_json(i18n_dict, output_file)
