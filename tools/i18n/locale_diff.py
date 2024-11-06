import json

def locale_diff(template, target):
    with open(template, 'r', encoding='utf-8') as f:
        template_key = json.load(f)
    with open(target, 'r', encoding='utf-8') as f:
        target_key = json.load(f)
    for key in template_key:
        if key in target_key:
            template_key[key] = target_key[key]
        else:
            print("Missing: " + key)
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(template_key, f, ensure_ascii=False, indent=4)

def sort(target):
    old = {}
    new = {}

    with open(target, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for key in data:
        if data[key] == "text":
            new[key] = data[key]
        else:
            old[key] = data[key]
    # 将old放前面，new放后面
    data = {**old, **new}
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

template = "locale/template.json"
target = ["locale/en_US.json", "locale/ja_JP.json", "locale/emoji.json"]
for t in target:
    locale_diff(template, t)
    sort(t)