<div align="center">

# How to use MSST WebUI-for-Clouds

</div>

> [!NOTE]
> 1. When used on cloud, you don't need to download the model yourself. WebUI will automatically download the model you need. 
> 2. If you want to modify the inference parameters of the preset process, edit the `data/webui_config.json` file.
> 3. The jupyter notebook for Google Colab is [here](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb)!

- Make sure you have Python 3.10 and ffmpeg, if not, install them first.
```bash
sudo apt install python3.10 ffmpeg
```

- Clone repository and install dependencies.
```bash
# Clone the repository
git clone https://github.com/SUC-DriverOld/MSST-WebUI.git
cd MSST-WebUI

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Install torch and other requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip uninstall librosa -y
pip install tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl

# Copy cloud webui files to the root directory
!cp tools/webUI_for_clouds/webUI_for_clouds.py .
!cp tools/webUI_for_clouds/download_models.py .
```

- If you want to use "Vocals to MIDI", use the following command to download the weights.
```bash
wget https://huggingface.co/Sucial/MSST-WebUI/resolve/main/SOME_weights/model_steps_64000_simplified.ckpt -O MSST-WebUI/tools/SOME_weights/model_steps_64000_simplified.ckpt
```

- You need to setup for WebUI manually. If you use jupyter, you can use the following code.
```python
# The modifiable parameters are as follows
LANGUAGE = "English" #@param ["Auto", "English", "简体中文", "繁體中文", "日本語", "😊"]
MODEL_DOWNLOAD_LINK = "huggingface.co" #@param ["Auto", "huggingface.co", "hf-mirror.com"]
DEBUG = False #@param {type:"boolean"}

import json
import os
import shutil

# Supported languages
language_dict = {"Auto": "Auto", "简体中文": "zh_CN", "繁體中文": "zh_TW", "English": "en_US", "日本語": "ja_JP", "😊": "emoji"}

# Make dirs if not exist
if not os.path.exists("data"):
    shutil.copytree("data_backup", "data")
if not os.path.exists("configs"):
    shutil.copytree("configs_backup", "configs")
os.makedirs("input", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Modify config file
with open("data/webui_config.json", 'r', encoding="utf-8") as f:
    config = json.load(f)
    config['settings']['language'] = language_dict[LANGUAGE]
    config['settings']['download_link'] = MODEL_DOWNLOAD_LINK
    config['settings']['debug'] = DEBUG
with open("data/webui_config.json", 'w', encoding="utf-8") as f:
    json.dump(config, f, indent=4)
```

- Run the webUI. After running, you can use the public links shown bellow to connect to the webUI.

```bash
!python webUI_for_clouds.py
```
