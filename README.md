<div align="center"><img src="docs/logo.png" alt="logo" width="128" height="128"></div>
<h1 align="center">MSST-WebUI</h1>
<div align="center">

English | [简体中文](docs/README_zh.md) | 繁體中文 | 日本語 | 한국어<br>
A WebUI app for Music-Source-Separation-Training and we packed UVR together!<br>
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb)
[![GitHub release](https://img.shields.io/github/v/release/SUC-DriverOld/MSST-WebUI?label=Version)](https://github.com/SUC-DriverOld/MSST-WebUI/releases/latest)
[![GitHub stars](https://img.shields.io/github/stars/SUC-DriverOld/MSST-WebUI?label=Stars&color=blue&style=flat)](https://github.com/SUC-DriverOld/MSST-WebUI/stargazers)
[![GitHub license](https://img.shields.io/github/license/SUC-DriverOld/MSST-WebUI?label=License)](https://github.com/SUC-DriverOld/MSST-WebUI/blob/main/LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Models-blue?)](https://huggingface.co/Sucial/MSST-WebUI)
<br>

</div>

## Introduction

This is a webUI for [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training), which is a repository for training models for music source separation. You can use this webUI to infer the MSST model and VR Models, and the preset process page allows you to customize the processing flow yourself. You can install models in the "Install Models" interface. If you have downloaded [Ultimate Vocal Remover (UVR)](https://github.com/Anjok07/ultimatevocalremovergui) before, you do not need to download VR Models again. You can go to the "Settings" page and directly select your UVR5 model folder. We also provide some convenient tools in the WebUI such as [Singing-Oriented MIDI Extractor (SOME)](https://github.com/openvpi/SOME/), advanced ensemble mode, and more.

## Usage

- **Windows**: Download the installer from [Releases](https://github.com/SUC-DriverOld/MSST-WebUI/releases) or clone this repository and run from source.<br>
- **Linux/macOS**: Clone this repository and run from source.<br>
- **Google Colab**: [Click here](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb) to run the webUI on Google Colab.

### Download links

|    Websites     | Download Links                                       | Notes                              |
|:---------------:|:----------------------------------------------------:|:----------------------------------:|
| Github Releases | https://github.com/SUC-DriverOld/MSST-WebUI/releases |      Only installer, no models     |
|   Huggingface   |  https://huggingface.co/Sucial/MSST-WebUI/tree/main  | Installer and all available models |

### Documents

We provided some detailed chinese documents for chinese users, click [here](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc) to jump. For other users, go to `docs` folder to find some documents. You can also see [deton24](https://github.com/deton24)'s [Instrumental and vocal & stems separation & mastering guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c), which is a great guide too.

## Run from source

- Clone this repository.

  ```bash
  git clone https://github.com/SUC-DriverOld/MSST-WebUI.git
  cd MSST-WebUI
  ```

- Create Python environment and install the requirements. We recommend to use python 3.10.

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

- For more detailed launching arguments, or if you want to run the webUI on a cloud platform, see [this document](docs/webui.md).

- You may meet the following error when using model_type swin_upernet: `ValueError: Make sure that the channel dimension of the pixel values match with the one set in the configuration`. Refer to [this issue](https://github.com/SUC-DriverOld/MSST-WebUI/issues/24) to solve the problem.

## CLI & Python API

See [this document](docs/inference.md) for more details.

## Training & Validation

See [this document](docs/training.md) for more details.

## Reference

- [deton24's Documents] [Instrumental and vocal & stems separation & mastering guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c)
- [KitsuneX07's ComfyMSS] [ComfyMSS](https://github.com/KitsuneX07/ComfyMSS)
- [PyQt-Fluent-Widgets] [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
- [Pyside6] [PySide6 documents](https://doc.qt.io/qtforpython-6)
- [python-audio-separator] [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [Singing-Oriented MIDI Extractor] [SOME](https://github.com/openvpi/SOME/)
- [Ultimate Vocal Remover] [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [ZFTurbo's MSST code] [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)

### Thanks to all contributors for their efforts

<a href="https://github.com/SUC-DriverOld/MSST-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=SUC-DriverOld/MSST-WebUI" alt=""/>
</a>

### Thanks for all model providers

<table>
  <tr>
    <td style="text-align: center;"><img src="https://github.com/anvuew.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="anvuew"></td>
    <td style="text-align: center;"><img src="https://github.com/aufr33.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="aufr33"></td>
    <td style="text-align: center;"><img src="https://github.com/deton24.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="deton24"></td>
    <td style="text-align: center;"><img src="https://github.com/jarredou.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="jarredou"></td>
    <td style="text-align: center;"><img src="https://github.com/pcunwa.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="pcunwa"></td>
    <td style="text-align: center;"><img src="https://github.com/SUC-DriverOld.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="Sucial"></td>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="https://github.com/anvuew">anvuew</a></td>
    <td style="text-align: center;"><a href="https://github.com/aufr33">aufr33</a></td>
    <td style="text-align: center;"><a href="https://github.com/deton24">deton24</a></td>
    <td style="text-align: center;"><a href="https://github.com/jarredou">jarredou</a></td>
    <td style="text-align: center;"><a href="https://github.com/pcunwa">pcunwa</a></td>
    <td style="text-align: center;"><a href="https://github.com/SUC-DriverOld">Sucial</a></td>
  </tr>
  <tr>
    <td style="text-align: center;"><img src="https://github.com/Super-YH.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="Super-YH"></td>
    <td style="text-align: center;"><img src="https://github.com/playdasegunda.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="viperx"></td>
    <td style="text-align: center;"><img src="https://github.com/wesleyr36.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="wesleyr36"></td>
    <td style="text-align: center;"><img src="https://github.com/yxlllc.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="yxlllc"></td>
    <td style="text-align: center;"><img src="https://github.com/ZFTurbo.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="ZFturbo"></td>
    <td></td>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="https://github.com/Super-YH">Super-YH</a></td>
    <td style="text-align: center;"><a href="https://github.com/playdasegunda">viperx</a></td>
    <td style="text-align: center;"><a href="https://github.com/wesleyr36">wesleyr36</a></td>
    <td style="text-align: center;"><a href="https://github.com/yxlllc">yxlllc</a></td>
    <td style="text-align: center;"><a href="https://github.com/ZFTurbo">ZFturbo</a></td>
    <td></td>
  </tr>
</table>
