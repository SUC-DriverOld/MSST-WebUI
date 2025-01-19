## WebUI Usage

Use `webui.py` to start the webui. There are two modes: local mode and cloud mode.

### Options

```
usage: webUI.py [-h] [-d] [--factory_reset] [-i IP_ADDRESS] [-p PORT] [-s] [--ues_cloud] [--language {Auto,zh_CN,zh_TW,en_US,ja_JP,ko_KR}] [--model_download_link {Auto,huggingface.co,hf-mirror.com}]

WebUI for Music Source Separation Training

options:
  -h, --help                                                show this help message and exit
  -d, --debug                                               Enable debug mode.
  --factory_reset                                           Reset WebUI settings and model seetings to default, clear cache and exit.

Local Startup Parameters:
  -i IP_ADDRESS, --ip_address IP_ADDRESS                    Server IP address (Default: Auto).
  -p PORT, --port PORT                                      Server port (Default: Auto).
  -s, --share                                               Enable share link.

Cloud Startup Parameters:
  --use_cloud                                               Enable cloud mode. When using in cloud platforms, enable this option.
  --language {Auto,zh_CN,zh_TW,en_US,ja_JP,ko_KR}           Set cloud WebUI language (Default: Auto).
  --model_download_link {Auto,huggingface.co,hf-mirror.com}
                                                            Set cloud model download link (Default: Auto).
```

To Factory Reset the WebUI settings and model settings to default, clear the cache and exit, use the following command:

```bash
python webui.py --factory_reset
```

### Local Mode

For those who want to run the webui on their own PC, you can simply use the following command to start the webui:

```bash
python webui.py
```

If you want to specify the IP address and port or you want to enable share link (gradio povided, expired in 72 hours), you can use the following command:

```bash
python webui.py -i 192.168.1.100 -p 7860 -s
```

### Cloud Mode

For those who want to run the webui on cloud platforms, you can use the following command to start the webui:

```bash
python webui.py --use_cloud
```

If you want to specify the language and model download link, you can use the following command:

```bash
python webui.py --use_cloud --language zh_CN --model_download_link hf-mirror.com
```

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

language_dict = {"Auto": "Auto", "简体中文": "zh_CN", "繁體中文": "zh_TW", "English": "en_US", "日本語": "ja_JP", "😊": "emoji", "한국어": "ko_KR"}
language = language_dict[LANGUAGE]
debug = "--debug" if DEBUG else ""

# using cloud mode to start webui
!python webUI.py --use_cloud --language {language} --model_download_link {MODEL_DOWNLOAD_LINK} {debug}
```
