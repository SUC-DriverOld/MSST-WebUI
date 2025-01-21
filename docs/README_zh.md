<div align="center"><img src="logo.png" alt="logo" width="128" height="128"></div>
<h1 align="center">MSST-WebUI</h1>
<div align="center">

[English](../README.md) | 简体中文 | 繁體中文 | 日本語 | 한국어<br>
一个为Music-Source-Separation-Training打造的WebUI应用，此外我们也添加了UVR进来！<br>
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb)
[![GitHub release](https://img.shields.io/github/v/release/SUC-DriverOld/MSST-WebUI?label=Version)](https://github.com/SUC-DriverOld/MSST-WebUI/releases/latest)
[![GitHub stars](https://img.shields.io/github/stars/SUC-DriverOld/MSST-WebUI?label=Stars&color=blue&style=flat)](https://github.com/SUC-DriverOld/MSST-WebUI/stargazers)
[![GitHub license](https://img.shields.io/github/license/SUC-DriverOld/MSST-WebUI?label=License)](https://github.com/SUC-DriverOld/MSST-WebUI/blob/main/LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Models-blue?)](https://huggingface.co/Sucial/MSST-WebUI)
<br>

</div>

## 介绍

这是一个用于 [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training) 的WebUI，MSST是一个用于训练音乐源分离模型的仓库。您可以使用这个WebUI推断MSST模型和VR模型，预设流程页面允许您自定义处理流程。您可以在“安装模型”界面安装模型。如果您之前已经下载了 [Ultimate Vocal Remover (UVR)](https://github.com/Anjok07/ultimatevocalremovergui)，则无需重新下载VR模型。您可以直接进入“设置”页面，选择您的UVR5模型文件夹。我们还在WebUI中提供了一些便捷的工具，例如 [Singing-Oriented MIDI Extractor (SOME)](https://github.com/openvpi/SOME/)、更高级的合奏模式等。

## 使用

- **Windows**: 从 [Releases](https://github.com/SUC-DriverOld/MSST-WebUI/releases) 下载安装程序，或克隆此仓库并从源代码运行。<br>
- **Linux/macOS**: 克隆此仓库并从源代码运行。<br>
- **Google Colab**: [点击这里](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb) 在Google Colab上运行WebUI。<br>

### 下载链接

推荐国内用户使用后三个下载链接，无需魔法即可下载。此外，我们向国内用户提供详细的[飞书教程文档](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)。

|    网站     | 下载链接                                          | 提取码   | 备注                                 |
|:-----------:|:-----------------------------------------------:|:--------:|:----------------------------------:|
| Github Releases | https://github.com/SUC-DriverOld/MSST-WebUI/releases |    -     | 仅提供安装程序，无模型下载  |
|   Huggingface   | https://huggingface.co/Sucial/MSST-WebUI/tree/main   |    -     | 安装程序及所有可用模型  |
| Huggingface镜像站 | https://hf-mirror.com/Sucial/MSST-WebUI/tree/main |    -     | 安装程序及所有可用模型  |
| 百度网盘 | https://pan.baidu.com/s/1uzYHSpMJ1nZVjRpIXIFF_Q  |   1145   | 安装程序及所有可用模型  |
| 123网盘  | https://www.123pan.cn/s/1bmETd-AefWh.html |   1145   | 安装程序及所有可用模型  |

### 文档

我们为中文用户提供了一些详细的中文文档，点击 [这里](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc) 跳转。对于其他语言的用户，前往 `docs` 文件夹查看部分文档。你也可以参考 [deton24](https://github.com/deton24) 的 [Instrumental and vocal & stems separation & mastering guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c)。

## 从源码运行

- 克隆仓库

  ```bash
  git clone https://github.com/SUC-DriverOld/MSST-WebUI.git
  cd MSST-WebUI
  ```

- 使用Conda创建Python虚拟环境，此处建议使用Python 3.10。

  ```bash
  conda create -n msst python=3.10 -y
  conda activate msst
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt --only-binary=samplerate
  ```

- 安装完依赖后，进入 `site-packages` 文件夹，打开 `librosa\util\utils.py` 文件，定位到第 **2185** 行。将该行 `np.dtype(complex): np.dtype(np.float).type,` 修改为 `np.dtype(complex): np.dtype(float).type,`。如果您不知道如何操作，可以使用以下命令。

  ```bash
  pip uninstall librosa -y
  pip install tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl
  ```

- 使用下面的命令启动WebUI。

  ```bash
  python webUI.py
  ```

- 如需更详细的启动参数，或如果您想在云平台上运行WebUI，请查看 [此文档(英文)](webui.md)。

- 使用模型类型 swin_upernet 时，您可能会遇到以下错误：`ValueError: Make sure that the channel dimension of the pixel values match with the one set in the configuration`。请参考 [此issue](https://github.com/SUC-DriverOld/MSST-WebUI/issues/24) 解决。

## CLI & Python API

参考 [此文档(英文)](inference.md) 获取更多详细信息。

## 训练 & 验证

参考 [此文档(英文)](training.md) 获取更多详细信息。

## 参考

- [deton24's Documents] [Instrumental and vocal & stems separation & mastering guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c)
- [KitsuneX07's ComfyMSS] [ComfyMSS](https://github.com/KitsuneX07/ComfyMSS)
- [PyQt-Fluent-Widgets] [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
- [Pyside6] [PySide6 documents](https://doc.qt.io/qtforpython-6)
- [python-audio-separator] [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [Singing-Oriented MIDI Extractor] [SOME](https://github.com/openvpi/SOME/)
- [Ultimate Vocal Remover] [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [ZFTurbo's MSST code] [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)

### 感谢所有贡献者的共同努力

<a href="https://github.com/SUC-DriverOld/MSST-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=SUC-DriverOld/MSST-WebUI" alt=""/>
</a>

### 感谢所有的模型提供者

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
