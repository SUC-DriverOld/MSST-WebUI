import opencc
import json

cc = opencc.OpenCC('s2t')

with open("locale/template.json", 'r', encoding='utf-8') as f:
    template = json.load(f)
    for key in template:
        template[key] = cc.convert(key)

with open("locale/zh_TW.json", 'w', encoding='utf-8') as f:
    json.dump(template, f, ensure_ascii=False, indent=4)
with open("locale/zh_HK.json", 'w', encoding='utf-8') as f:
    json.dump(template, f, ensure_ascii=False, indent=4)
with open("locale/zh_SG.json", 'w', encoding='utf-8') as f:
    json.dump(template, f, ensure_ascii=False, indent=4)