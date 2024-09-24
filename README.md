<div align="center">

# MSST-WebUI
[![Try it on Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb)
[![GitHub release](https://img.shields.io/github/v/release/SUC-DriverOld/MSST-WebUI)](https://github.com/SUC-DriverOld/MSST-WebUI/releases/latest)
[![GitHub license](https://img.shields.io/github/license/SUC-DriverOld/MSST-WebUI)](https://github.com/SUC-DriverOld/MSST-WebUI/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/SUC-DriverOld/MSST-WebUI)](https://github.com/SUC-DriverOld/MSST-WebUI/stargazers)<br>
WebUI of Music-Source-Separation-Training-Inference , and we packed UVR together!<br>
English✅ | 简体中文✅ | 繁體中文✅ | 日本語✅
</div>

## Introduction

This is a webUI for [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training), which is a music source separation training framework. You can use this webUI to infer the MSST model and UVR VR.Models (The inference code comes from [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), and we do some changes on it), and the preset process page allows you to customize the processing flow yourself. You can install models in the "Install Models" interface. If you have downloaded [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) before, you do not need to download VR.Models again. You can go to the "Settings" page and directly select your UVR5 model folder. Finally, we also provide some convenient tools such as [SOME: Vocals to MIDI](https://github.com/openvpi/SOME/) in the webUI.

## Usage

**Windows**: Download the installer from [Releases](https://github.com/SUC-DriverOld/MSST-WebUI/releases) and run it. Or you can clone this repository and run from source.<br>
**Linux/MacOS**: Clone this repository and run from source.<br>
**Google Colab**: [Click here](https://colab.research.google.com/github/SUC-DriverOld/MSST-WebUI/blob/main/webUI_for_colab.ipynb) to run the webUI on Google Colab.

### 中国用户可以从下方链接下载安装包

下载地址：[123云盘](https://www.123pan.com/s/1bmETd-AefWh.html) 提取码: 1145 | [百度网盘](https://pan.baidu.com/s/1uzYHSpMJ1nZVjRpIXIFF_Q?pwd=1145) 提取码: 1145<br>
相关使用教程： [B站教程视频](https://www.bilibili.com/video/BV18m42137rm) | [飞书教程文档](https://r1kc63iz15l.feishu.cn/wiki/JSp3wk7zuinvIXkIqSUcCXY1nKc)（视频随时落后，文档保持更新）

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
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
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

- If you run webUI on a cloud platform, see [this document](tools/webUI_for_clouds/README.md) for more details.

> [!NOTE]
> When using model_type `swin_upernet`, you may meet the following error:
> ```bash
> ValueError: Make sure that the channel dimension of the pixel values match with the one set in the configuration.
> ```
> Please refer to [this issue](https://github.com/SUC-DriverOld/MSST-WebUI/issues/24) to solve the problem.

## Command Line

### MSST Inference

Use `Inference.py`

```bash
usage: msst_inference.py [-h] [--model_type MODEL_TYPE] [--config_path CONFIG_PATH] [--start_check_point START_CHECK_POINT] [--input_folder INPUT_FOLDER]
                         [--output_format OUTPUT_FORMAT] [--store_dir STORE_DIR] [--device_ids DEVICE_IDS [DEVICE_IDS ...]] [--extract_instrumental]
                         [--instrumental_only] [--extra_store_dir EXTRA_STORE_DIR] [--force_cpu] [--use_tta]

options:
  -h, --help                                show this help message and exit
  --model_type MODEL_TYPE                   One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer, scnet, scnet_unofficial, segm_models, swin_upernet, torchseg
  --config_path CONFIG_PATH                 path to config file
  --start_check_point START_CHECK_POINT     Initial checkpoint to valid weights
  --input_folder INPUT_FOLDER               folder with mixtures to process
  --output_format OUTPUT_FORMAT             output format for separated files, one of wav, flac, mp3
  --store_dir STORE_DIR                     path to store results files
  --device_ids DEVICE_IDS [DEVICE_IDS ...]  list of gpu ids
  --extract_instrumental                    invert vocals to get instrumental if provided
  --instrumental_only                       extract instrumental only
  --extra_store_dir EXTRA_STORE_DIR         path to store extracted instrumental. If not provided, store_dir will be used
  --force_cpu                               Force the use of CPU even if CUDA is available
  --use_tta                                 Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality.
```

### VR Inference

Use `uvr_inference.py`

> [!NOTE]
> 1. Only `VR_Models` can be used for UVR Inference. You can use other models like MDX23C models and HTDemucs models in MSST Inference.<br>
> 2. We do some changes on the code and now you can import folder_path for UVR Inference!

```bash
usage: uvr_inference.py [-h] [-d] [-m MODEL_FILENAME] [--output_format OUTPUT_FORMAT] [--output_dir OUTPUT_DIR] [--model_file_dir MODEL_FILE_DIR]
                        [--extra_output_dir EXTRA_OUTPUT_DIR] [--invert_spect] [--normalization NORMALIZATION] [--single_stem SINGLE_STEM] [--use_cpu] [--save_another_stem]
                        [--vr_batch_size VR_BATCH_SIZE] [--vr_window_size VR_WINDOW_SIZE] [--vr_aggression VR_AGGRESSION] [--vr_enable_tta] [--vr_high_end_process]
                        [--vr_enable_post_process] [--vr_post_process_threshold VR_POST_PROCESS_THRESHOLD]
                        [audio_file]

Separate audio file into different stems.

positional arguments:
  audio_file                                             The audio file path to separate, in any common format. You can input file path or file folder path

options:
  -h, --help                                             show this help message and exit
  -d, --debug                                            Enable debug logging, equivalent to --log_level=debug.

Separation I/O Params:
  -m MODEL_FILENAME, --model_filename MODEL_FILENAME     model to use for separation (default: 1_HP-UVR.pth). Example: -m 2_HP-UVR.pth
  --output_format OUTPUT_FORMAT                          output format for separated files, any common format (default: FLAC). Example: --output_format=MP3
  --output_dir OUTPUT_DIR                                directory to write output files (default: <current dir>). Example: --output_dir=/app/separated
  --model_file_dir MODEL_FILE_DIR                        model files directory (default: pretrain/VR_Models). Example: --model_file_dir=/app/models
  --extra_output_dir EXTRA_OUTPUT_DIR                    extra output directory for saving another stem. If not provided, output_dir will be used. Example: --extra_output_dir=/app/extra_output

Common Separation Parameters:
  --invert_spect                                         invert secondary stem using spectogram (default: False). Example: --invert_spect
  --normalization NORMALIZATION                          max peak amplitude to normalize input and output audio to (default: 0.9). Example: --normalization=0.7
  --single_stem SINGLE_STEM                              output only single stem, e.g. Instrumental, Vocals, Drums, Bass, Guitar, Piano, Other. Example: --single_stem=Instrumental   
  --use_cpu                                              use CPU instead of GPU for inference
  --save_another_stem                                    save another stem when using flow inference (default: False). Example: --save_another_stem

VR Architecture Parameters:
  --vr_batch_size VR_BATCH_SIZE                          number of batches to process at a time. higher = more RAM, slightly faster processing (default: 4). Example: --vr_batch_size=16
  --vr_window_size VR_WINDOW_SIZE                        balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: 512). Example: --vr_window_size=320
  --vr_aggression VR_AGGRESSION                          intensity of primary stem extraction, -100 - 100. typically 5 for vocals & instrumentals (default: 5). Example: --vr_aggression=2
  --vr_enable_tta                                        enable Test-Time-Augmentation; slow but improves quality (default: False). Example: --vr_enable_tta
  --vr_high_end_process                                  mirror the missing frequency range of the output (default: False). Example: --vr_high_end_process
  --vr_enable_post_process                               identify leftover artifacts within vocal output; may improve separation for some songs (default: False). Example: --vr_enable_post_process
  --vr_post_process_threshold VR_POST_PROCESS_THRESHOLD  threshold for post_process feature: 0.1-0.3 (default: 0.2). Example: --vr_post_process_threshold=0.1
```

### MSST Training

Use `train.py`. If you use multi-GPUs, try to use `train_accelerate.py`. But it's still under experiment.

```bash
usage: train_accelerate.py [-h] [--model_type MODEL_TYPE] [--config_path CONFIG_PATH] [--start_check_point START_CHECK_POINT] [--results_path RESULTS_PATH]
                           [--data_path DATA_PATH [DATA_PATH ...]] [--dataset_type DATASET_TYPE] [--valid_path VALID_PATH [VALID_PATH ...]] [--num_workers NUM_WORKERS]
                           [--pin_memory] [--seed SEED] [--device_ids DEVICE_IDS [DEVICE_IDS ...]] [--use_multistft_loss] [--use_mse_loss] [--use_l1_loss] [--pre_valid]

options:
  -h, --help                                show this help message and exit
  --model_type MODEL_TYPE                   One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit
  --config_path CONFIG_PATH                 path to config file
  --start_check_point START_CHECK_POINT     Initial checkpoint to start training
  --results_path RESULTS_PATH               path to folder where results will be stored (weights, metadata)
  --data_path DATA_PATH [DATA_PATH ...]     Dataset data paths. You can provide several folders.
  --dataset_type DATASET_TYPE               Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md
  --valid_path VALID_PATH [VALID_PATH ...]  validation data paths. You can provide several folders.
  --num_workers NUM_WORKERS                 dataloader num_workers
  --pin_memory                              dataloader pin_memory
  --seed SEED                               random seed
  --device_ids DEVICE_IDS [DEVICE_IDS ...]  list of gpu ids
  --use_multistft_loss                      Use MultiSTFT Loss (from auraloss package)
  --use_mse_loss                            Use default MSE loss
  --use_l1_loss                             Use L1 loss
  --pre_valid                               Run validation before training
```

### Thanks

- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [SOME: Vocals to MIDI](https://github.com/openvpi/SOME/)
- [@KitsuneX07](https://github.com/KitsuneX07) | [bilibili@阿狸不吃隼舞](https://space.bilibili.com/403335715)
- [@SUC-DriverOld](https://github.com/SUC-DriverOld) | [Bilibili@Sucial丶](https://space.bilibili.com/445022409)
