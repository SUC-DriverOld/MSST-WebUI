<div align="center">

# MSST-WebUI

Music Source Separation Training Inference Webui, besides, we packed UVR together!
</div>

## Introduction

This is a webUI for [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training), which is a music source separation training framework. You can use this webUI to infer the MSST model and UVR VR.Models (The inference code comes from [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), and we do some changes on it), and the preset process page allows you to customize the processing flow yourself. You can install models in the "Install Models" interface. If you have downloaded [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) before, you do not need to download VR.Models again. You can go to the "Settings" page and directly select your UVR5 model folder. Finally, we also provide four convenient tools in the webUI.

## Usage

### You can go to [Releases](https://github.com/SUC-DriverOld/MSST-WebUI/releases) to download the installer

### 中国用户可以从下方链接下载

下载地址：https://www.123pan.com/s/1bmETd-AefWh.html 提取码:1145<br>
B站教程视频：https://www.bilibili.com/video/BV18m42137rm

## Run from source

1. Clone this repository.
2. Create Python environment and install the requirements.

   ```bash
   conda create -n msst python=3.10 -y
   conda activate msst
   pip install -r requirements.txt
   ```

> [!NOTE]
> 1. You may meet some problems when using UVR-Separate, they comes from dependances Numpy and Librosa. You may need to change the code `np.float` to `np.float64` or `np.float32` according to the traceback message.
> 2. During install the requirements, you may meet conflict between huggingface-Hub and gradio. Just ignore it, it will not affect the use of the webUI.

3. Run the webui use the following command.

   ```bash
   python webUI.py
   ```

## Command Line

### MSST Inference

Use `Inference.py`

```bash
usage: inference.py [-h] [--model_type MODEL_TYPE] [--config_path CONFIG_PATH] [--start_check_point START_CHECK_POINT] [--input_folder INPUT_FOLDER] [--store_dir STORE_DIR]
                    [--device_ids DEVICE_IDS [DEVICE_IDS ...]] [--extract_instrumental] [--force_cpu]

options:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit
  --config_path CONFIG_PATH
                        path to config file
  --start_check_point START_CHECK_POINT
                        Initial checkpoint to valid weights
  --input_folder INPUT_FOLDER
                        folder with mixtures to process
  --store_dir STORE_DIR
                        path to store results as wav file
  --device_ids DEVICE_IDS [DEVICE_IDS ...]
                        list of gpu ids
  --extract_instrumental
                        invert vocals to get instrumental if provided
  --force_cpu           Force the use of CPU even if CUDA is available
```

### UVR Inference

Use `uvr_inference.py`

> [!NOTE]
> Only VR_Models can be used for UVR Inference.
> You can use other models like MDX23C models and HTDemucs models in MSST Inference.
> Fix: You can now import folder_path for UVR Inference!

```bash
usage: uvr_inference.py [-h] [-d] [-e] [-l] [--log_level LOG_LEVEL] [-m MODEL_FILENAME] 
                        [--output_format OUTPUT_FORMAT] [--output_dir OUTPUT_DIR] [--model_file_dir MODEL_FILE_DIR] 
                        [--invert_spect] [--normalization NORMALIZATION] [--single_stem SINGLE_STEM] [--sample_rate SAMPLE_RATE] [--use_cpu]
                        [--vr_batch_size VR_BATCH_SIZE] [--vr_window_size VR_WINDOW_SIZE] [--vr_aggression VR_AGGRESSION] [--vr_enable_tta] [--vr_high_end_process] [--vr_enable_post_process]
                        [--vr_post_process_threshold VR_POST_PROCESS_THRESHOLD] 
                        [audio_file]

Separate audio file into different stems.

positional arguments:
  audio_file                                             The audio file path to separate, in any common format. You can input file path or file folder path

options:
  -h, --help                                             show this help message and exit

Info and Debugging:
  -d, --debug                                            Enable debug logging, equivalent to --log_level=debug.
  -e, --env_info                                         Print environment information and exit.
  -l, --list_models                                      List all supported models and exit.
  --log_level LOG_LEVEL                                  Log level, e.g. info, debug, warning (default: info).

Separation I/O Params:
  -m MODEL_FILENAME, --model_filename MODEL_FILENAME     model to use for separation (default: model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt). Example: -m 2_HP-UVR.pth
  --output_format OUTPUT_FORMAT                          output format for separated files, any common format (default: FLAC). Example: --output_format=MP3
  --output_dir OUTPUT_DIR                                directory to write output files (default: <current dir>). Example: --output_dir=/app/separated
  --model_file_dir MODEL_FILE_DIR                        model files directory (default: /tmp/audio-separator-models/). Example: --model_file_dir=/app/models

Common Separation Parameters:
  --invert_spect                                         invert secondary stem using spectogram (default: False). Example: --invert_spect
  --normalization NORMALIZATION                          max peak amplitude to normalize input and output audio to (default: 0.9). Example: --normalization=0.7
  --single_stem SINGLE_STEM                              output only single stem, e.g. Instrumental, Vocals, Drums, Bass, Guitar, Piano, Other. Example: --single_stem=Instrumental
  --sample_rate SAMPLE_RATE                              modify the sample rate of the output audio (default: 44100). Example: --sample_rate=44100
  --use_cpu                                              use CPU instead of GPU for inference

VR Architecture Parameters:
  --vr_batch_size VR_BATCH_SIZE                          number of batches to process at a time. higher = more RAM, slightly faster processing (default: 4). Example: --vr_batch_size=16        
  --vr_window_size VR_WINDOW_SIZE                        balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: 512). Example: --vr_window_size=320       
  --vr_aggression VR_AGGRESSION                          intensity of primary stem extraction, -100 - 100. typically 5 for vocals & instrumentals (default: 5). Example: --vr_aggression=2      
  --vr_enable_tta                                        enable Test-Time-Augmentation; slow but improves quality (default: False). Example: --vr_enable_tta
  --vr_high_end_process                                  mirror the missing frequency range of the output (default: False). Example: --vr_high_end_process
  --vr_enable_post_process                               identify leftover artifacts within vocal output; may improve separation for some songs (default: False). Example: --vr_enable_post_process
  --vr_post_process_threshold VR_POST_PROCESS_THRESHOLD  threshold for post_process feature: 0.1-0.3 (default: 0.2). Example: --vr_post_process_threshold=0.1
```

### Train MSST

Use `train.py`

```bash
usage: train.py [-h] [--model_type MODEL_TYPE] [--config_path CONFIG_PATH] [--start_check_point START_CHECK_POINT] [--results_path RESULTS_PATH] [--data_path DATA_PATH [DATA_PATH ...]]
                [--dataset_type DATASET_TYPE] [--valid_path VALID_PATH [VALID_PATH ...]] [--num_workers NUM_WORKERS] [--pin_memory PIN_MEMORY] [--seed SEED]
                [--device_ids DEVICE_IDS [DEVICE_IDS ...]] [--use_multistft_loss] [--use_mse_loss] [--use_l1_loss]

options:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit
  --config_path CONFIG_PATH
                        path to config file
  --start_check_point START_CHECK_POINT
                        Initial checkpoint to start training
  --results_path RESULTS_PATH
                        path to folder where results will be stored (weights, metadata)
  --data_path DATA_PATH [DATA_PATH ...]
                        Dataset data paths. You can provide several folders.
  --dataset_type DATASET_TYPE
                        Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md
  --valid_path VALID_PATH [VALID_PATH ...]
                        validation data paths. You can provide several folders.
  --num_workers NUM_WORKERS
                        dataloader num_workers
  --pin_memory PIN_MEMORY
                        dataloader pin_memory
  --seed SEED           random seed
  --device_ids DEVICE_IDS [DEVICE_IDS ...]
                        list of gpu ids
  --use_multistft_loss  Use MultiSTFT Loss (from auraloss package)
  --use_mse_loss        Use default MSE loss
  --use_l1_loss         Use L1 loss
```

### Thanks

- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [@KitsuneX07](https://github.com/KitsuneX07) | [bilibili@阿狸不吃隼舞](https://space.bilibili.com/403335715)
- [@SUC-DriverOld](https://github.com/SUC-DriverOld) | [Bilibili@Sucial丶](https://space.bilibili.com/445022409)
