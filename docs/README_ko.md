<div align="center"><img src="logo.png" alt="logo" width="128" height="128"></div>
<h1 align="center">MSST-WebUI</h1>
<div align="center">

[English](../README.md) | [简体中文](README_zh.md) | 繁體中文 | 日本語 | 한국어<br>
Music-Source-Separation-Training용 WebUI 애플리케이션, UVR도 함께 포함되어 있습니다!<br>
[![GitHub release](https://img.shields.io/github/v/release/SUC-DriverOld/MSST-WebUI?label=Version)](https://github.com/SUC-DriverOld/MSST-WebUI/releases/latest) [![GitHub stars](https://img.shields.io/github/stars/SUC-DriverOld/MSST-WebUI?label=Stars&color=blue&style=flat)](https://github.com/SUC-DriverOld/MSST-WebUI/stargazers) [![GitHub license](https://img.shields.io/github/license/SUC-DriverOld/MSST-WebUI?label=License)](https://github.com/SUC-DriverOld/MSST-WebUI/blob/main/LICENSE) [![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Models-blue?)](https://huggingface.co/Sucial/MSST-WebUI)<br>
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb) [![UCloud](https://img.shields.io/badge/一键部署-优云智算UCloud-orange?)](https://www.compshare.cn/images-detail?ImageID=compshareImage-19o9qlm9x5f6&referral_code=1ywd4VqDKknFWCEUZvOoWo&ytag=GPU_Github_ZFTurbo)

</div>

## 소개

이 WebUI는 [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training)를 위한 사용자 인터페이스입니다. 이 저장소는 음악 소스 분리 모델을 학습하기 위한 도구들을 포함하고 있습니다. 본 WebUI를 사용하여 MSST 모델과 VR 모델을 추론할 수 있으며, 사전 설정된 프로세스 페이지에서 처리 과정을 직접 커스터마이징할 수 있습니다.

“Install Models” 메뉴에서 모델을 설치할 수 있으며, 기존에 [Ultimate Vocal Remover (UVR)](https://github.com/Anjok07/ultimatevocalremovergui)를 다운로드한 경우 VR 모델을 다시 받을 필요 없이 “Settings” 페이지에서 UVR5 모델 폴더를 선택하면 됩니다. 또한 [SOME (Singing-Oriented MIDI Extractor)](https://github.com/openvpi/SOME/)와 고급 앙상블 모드 등 다양한 편의 기능도 제공됩니다.

## 사용 방법

- **Windows**: [Releases](https://github.com/SUC-DriverOld/MSST-WebUI/releases)에서 설치 파일을 받거나 저장소를 클론하여 직접 실행합니다.<br>
- **Linux/macOS**: 저장소를 클론하고 직접 실행합니다.<br>
- **Google Colab**: [여기 클릭](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb)하여 Colab에서 실행합니다.
- **UCloud (중국 사용자 전용)**: [여기 클릭](https://www.compshare.cn/images-detail?ImageID=compshareImage-19o9qlm9x5f6&referral_code=1ywd4VqDKknFWCEUZvOoWo&ytag=GPU_Github_ZFTurbo)하여 UCloud에 배포합니다.

### 다운로드 링크

| 웹사이트         | 다운로드 링크                                             | 비고                           |
|------------------|----------------------------------------------------------|--------------------------------|
| Github Releases  | https://github.com/SUC-DriverOld/MSST-WebUI/releases     | 설치 파일만 포함, 모델 없음   |
| Huggingface      | https://huggingface.co/Sucial/MSST-WebUI/tree/main       | 설치 파일 및 모든 모델 포함   |

### 문서

중국어 문서는 [여기](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)를 클릭하세요. 그 외 사용자들은 `docs` 폴더에서 문서를 확인할 수 있습니다. 또한 [deton24](https://github.com/deton24)의 [보컬 및 악기 분리 & 마스터링 가이드](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c)도 매우 유용합니다.

## 소스 코드를 받아서 실행하기
* 저장소 클론
```bash
git clone https://github.com/SUC-DriverOld/MSST-WebUI.git
cd MSST-WebUI
```

* Python 환경 생성 및 라이브러리 설치 (Python 3.10 권장):

```bash
conda create -n msst python=3.10 -y
conda activate msst
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --only-binary=samplerate
```

* 라이브러리 설치 후, 요구사항 설치 후, site-packages 폴더로 이동하여 librosa\util\utils.py 파일을 열고 **2185번째 줄**로 이동합니다. 다음과 같이 코드를 수정하세요: `np.dtype(complex): np.dtype(np.float).type, → np.dtype(complex): np.dtype(float).type`,
만약 이 작업이 어렵거나 익숙하지 않다면, 아래 명령어를 사용해 쉽게 대체할 수 있습니다.

```bash
pip uninstall librosa -y
pip install tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl
```

* WebUI 실행:

```bash
python webUI.py
```

* 자세한 실행 인자나 클라우드 실행은 [이 문서](webui.md) 참고

* 모델 유형이 `swin_upernet`일 때 다음 오류가 발생할 수 있음: `ValueError: Make sure that the channel dimension of the pixel values match with the one set in the configuration`. [이슈 #24](https://github.com/SUC-DriverOld/MSST-WebUI/issues/24) 참고하세요

## CLI 및 Python API

[docs/inference.md](inference.md) 참고

## 학습 및 검증

[docs/training.md](training.md) 참고

## 참고 자료

- [deton24's Documents] [Instrumental and vocal & stems separation & mastering guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c)
- [KitsuneX07's ComfyMSS] [ComfyMSS](https://github.com/KitsuneX07/ComfyMSS)
- [PyQt-Fluent-Widgets] [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
- [Pyside6] [PySide6 documents](https://doc.qt.io/qtforpython-6)
- [python-audio-separator] [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [Singing-Oriented MIDI Extractor] [SOME](https://github.com/openvpi/SOME/)
- [Ultimate Vocal Remover] [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [ZFTurbo's MSST code] [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)

---

### 기여자 여러분께 감사드립니다

<a href="https://github.com/SUC-DriverOld/MSST-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=SUC-DriverOld/MSST-WebUI" alt=""/>
</a>

### 모델 제공자 분들께 감사드립니다

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
