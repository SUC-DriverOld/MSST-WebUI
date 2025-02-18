## WebUI Usage

Use `webui.py` to start the webui. There are two modes: local mode and cloud mode.

### Options

```
usage: webUI.py [-h] [-d] [-i IP_ADDRESS] [-p PORT] [-s] [--use_cloud] [--language {None,Auto,zh_CN,zh_TW,en_US,ja_JP,ko_KR}] [--model_download_link {None,Auto,huggingface.co,hf-mirror.com}]
                [--factory_reset]

WebUI for Music Source Separation Training

options:
  -h, --help                                                show this help message and exit
  -d, --debug                                               Enable debug mode.
  -i IP_ADDRESS, --ip_address IP_ADDRESS                    Server IP address (Default: Auto).
  -p PORT, --port PORT                                      Server port (Default: Auto).
  -s, --share                                               Enable share link.
  --use_cloud                                               Use special WebUI in cloud platforms.
  --language {None,Auto,zh_CN,zh_TW,en_US,ja_JP,ko_KR}      Set WebUI language (Default: Auto).
  --model_download_link {None,Auto,huggingface.co,hf-mirror.com}
                                                            Set model download link (Default: Auto).
  --factory_reset                                           Reset WebUI settings and model seetings to default, clear cache and exit.
```

- `-d` or `--debug`: Enable debug mode.
- `-i` or `--ip_address`: Server IP address (Default: Auto).
- `-p` or `--port`: Server port (Default: Auto).
- `-s` or `--share`: Use Gradio's intranet penetration service to penetrate WebUI to the public network. If your localhost cannot be accessed, please enable this option.
- `--use_cloud`: Use the special WebUI in cloud platforms. Having the functions of uploading files, packaging, and downloading.
- `--language`: Set the language of the cloud WebUI.
- `--model_download_link`: Set the model download link.
- `--factory_reset`: Reset the WebUI settings and model settings to default, clear the cache and exit.

## How to use MSST WebUI on clouds

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
```

- If you want to use "Vocals to MIDI", use the following command to download the weights.

```bash
wget https://huggingface.co/Sucial/MSST-WebUI/resolve/main/SOME_weights/model_steps_64000_simplified.ckpt -O MSST-WebUI/tools/SOME_weights/model_steps_64000_simplified.ckpt
```

- You need to setup for WebUI manually. If you use jupyter or colab, you can use the following code.

```python
LANGUAGE = "English" #@param ["Auto", "English", "简体中文", "繁體中文", "日本語", "😊", "한국어"]
MODEL_DOWNLOAD_LINK = "huggingface.co" #@param ["Auto", "huggingface.co", "hf-mirror.com"]
DEBUG = False #@param {type:"boolean"}
IP_ADDRESS = "0.0.0.0" #@param {type:"string"}
PORT = 7860 #@param {type:"integer"}

language_dict = {"Auto": "Auto", "简体中文": "zh_CN", "繁體中文": "zh_TW", "English": "en_US", "日本語": "ja_JP", "😊": "emoji", "한국어": "ko_KR"}
language = language_dict[LANGUAGE]
debug = "--debug" if DEBUG else ""

# Using cloud mode to start webui
!python webUI.py --use_cloud --share --language {language} --model_download_link {MODEL_DOWNLOAD_LINK} {debug} --ip_address {IP_ADDRESS} --port {PORT}
```
