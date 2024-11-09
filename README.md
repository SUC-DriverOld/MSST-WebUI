<div align="center">
  <img src="docs/logo.png" alt="logo" width="128" height="128">

# MSST-WebUI
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb)
[![GitHub release](https://img.shields.io/github/v/release/SUC-DriverOld/MSST-WebUI)](https://github.com/SUC-DriverOld/MSST-WebUI/releases/latest)
[![GitHub license](https://img.shields.io/github/license/SUC-DriverOld/MSST-WebUI)](https://github.com/SUC-DriverOld/MSST-WebUI/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/SUC-DriverOld/MSST-WebUI)](https://github.com/SUC-DriverOld/MSST-WebUI/stargazers)<br>
WebUI of Music-Source-Separation-Training-Inference , and we packed UVR together!<br>
Support Languages: English, 简体中文, 繁體中文, 日本語
</div>

## Introduction

This is a webUI for [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training), which is a music source separation training framework. You can use this webUI to infer the MSST model and VR Models (Inference code comes from [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), and we do some changes on it), and the preset process page allows you to customize the processing flow yourself. You can install models in the "Install Models" interface. If you have downloaded [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) before, you do not need to download VR Models again. You can go to the "Settings" page and directly select your UVR5 model folder. Finally, we also provide some convenient tools such as [SOME](https://github.com/openvpi/SOME/): Vocals to MIDI in the webUI.

## Usage

**Windows**: Download the installer from [Releases](https://github.com/SUC-DriverOld/MSST-WebUI/releases) and run it. Or you can clone this repository and run from source.<br>
**Linux/macOS**: Clone this repository and run from source.<br>
**Google Colab**: [Click here](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb) to run the webUI on Google Colab.<br>
**[For Chinese Users] Feishu Documents**：[Click to jump](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc?from=from_copylink)

### Available Download links

|    Websites     | Download Links                                       | Extract Code | Notes                              |
|:---------------:|------------------------------------------------------|:------------:|------------------------------------|
| Github Releases | https://github.com/SUC-DriverOld/MSST-WebUI/releases |      -       | Only installer, no models          |
|   Huggingface   | https://huggingface.co/Sucial/MSST-WebUI/tree/main   |      -       | Installer and all available models |
| [For Chinese Users] hf-mirror  | https://hf-mirror.com/Sucial/MSST-WebUI/tree/main    |      -       | Installer and all available models  |
| [For Chinese Users] BaiduNetdisk  | https://pan.baidu.com/s/1uzYHSpMJ1nZVjRpIXIFF_Q      |     1145     | Installer and all available models  |
| [For Chinese Users] 123Pan | https://www.123pan.cn/s/1bmETd-AefWh.html            |     1145     | Installer and all available models  |

## Run from source

- Clone this repository.

  ```bash
  git clone https://github.com/SUC-DriverOld/MSST-WebUI.git
  cd MSST-WebUI
  ```

- Create Python environment and install the requirements.

  ```bash
  conda create -n msst python=3.10 -y
  conda activate msst
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt --only-binary=samplerate
  ```
- After installing the requirements, go to `site-packages` folder, open `librosa\util\utils.py` and go to line **2185**. Change the line from `np.dtype(complex): np.dtype(np.float).type,` to `np.dtype(complex): np.dtype(float).type,`. If you do not know how to do this, you can use the following command.

  ```bash
  pip uninstall librosa -y
  pip install tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl
  ```

- Run the webUI use the following command.

  ```bash
  python webUI.py
  ```

  The optional arguments are as follows.

  ```
  usage: webUI.py [-h] [--server_name SERVER_NAME] [--server_port SERVER_PORT] [--share] [--debug]

  options:
    -h, --help                 show this help message and exit
    --server_name SERVER_NAME  Server IP address (Default: Auto). For example: 0.0.0.0
    --server_port SERVER_PORT  Server port (Default: Auto). For example: 7860
    --share                    Enable share link (Default: False).
    --debug                    Enable debug mode (Default: False).
  ```

- If you run webUI on a cloud platform, see [this document](tools/webUI_for_clouds/README.md) for more details.

> [!NOTE]
> When using model_type `swin_upernet`, you may meet the following error: `ValueError: Make sure that the channel dimension of the pixel values match with the one set in the configuration.`. Please refer to [this issue](https://github.com/SUC-DriverOld/MSST-WebUI/issues/24) to solve the problem.

## CLI & API

Please refer to [this document](docs/inference.md) for more details.

## Training

Please refer to [this document](docs/training.md) for more details.

## Reference

- [ZFTurbo's code] [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)<br>
- [python-audio-separator] [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)<br>
- [Ultimate Vocal Remover] [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)<br>
- [Vocals to MIDI] [SOME](https://github.com/openvpi/SOME/)<br>
- [@KitsuneX07] [Github](https://github.com/KitsuneX07) | [Bilibili](https://space.bilibili.com/403335715)<br>
- [@SUC-DriverOld] [Github](https://github.com/SUC-DriverOld) | [Bilibili](https://space.bilibili.com/445022409)

### Thanks to all contributors for their efforts

<a href="https://github.com/SUC-DriverOld/MSST-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=SUC-DriverOld/MSST-WebUI" alt=""/>
</a>
