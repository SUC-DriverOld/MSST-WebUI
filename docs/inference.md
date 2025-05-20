# Inference Usage

English | [简体中文](https://r1kc63iz15l.feishu.cn/wiki/WvcDwDfXniFzyzkhxPncV2SHnzc)

We provide two calling methods: API and CLI. For API calls, We provide two Python class interfaces for MSST separation and VR separation. For CLI, we have provided following CLI scripts in the `scripts` folder.

- [scripts/ensemble_audio_cli.py](../scripts/ensemble_audio_cli.py): Use multiple audio of the same duration for ensemble.
- [scripts/ensemble_cli.py](../scripts/ensemble_cli.py): Use multiple models to separate the audio first, and then ensemble the separated results.
- [scripts/msst_cli.py](../scripts/msst_cli.py): Use a single MSST model to separate audios.
- [scripts/preset_infer_cli.py](../scripts/preset_infer_cli.py): Use pre made presets for audio separation.
- [scripts/some_cli.py](../scripts/some_cli.py): Extract the MIDI file from imput audio.
- [scripts/vr_cli.py](../scripts/vr_cli.py): Use a single VR model to separate audios.

We use two JSON files as index files for the MSST model and VR model. They are located at `data_backup/msst_model_map.json` and `data_backup/vr_model_map.json`. The index files for third-party models added by users via the webUI are located at `config_unofficial/unofficial_msst_model.json` and `config_unofficial/unofficial_vr_model.json`. These two JSON files will only be created when the user adds a third-party model for the first time.

When using the API, you may need to manually input the separated stem names, which need to be obtained through these JSON index files. For the MSST model, the stem names are located in the model configuration file (*.yaml) under `config.training.instruments`. For the VR model, the stem names are directly located in the corresponding model name in the JSON index file, under `primary_stem` and `secondary_stem`.

## Contents

- [Inference Usage](#inference-usage)
  - [Contents](#contents)
  - [MSST API](#msst-api)
    - [Init parameters](#init-parameters)
    - [Functions](#functions)
  - [VR API](#vr-api)
    - [Init parameters](#init-parameters-1)
    - [Functions](#functions-1)
  - [msst\_cli.py](#msst_clipy)
  - [vr\_cli.py](#vr_clipy)
  - [preset\_infer\_cli.py](#preset_infer_clipy)
    - [Preset format](#preset-format)
  - [ensemble\_cli.py](#ensemble_clipy)
    - [Ensemble preset format](#ensemble-preset-format)
  - [ensemble\_audio\_cli.py](#ensemble_audio_clipy)
  - [some\_cli.py](#some_clipy)

## MSST API

Here is a simple class calling method.

```python
from inference.msst_infer import MSSeparator
from utils.logger import get_logger

separator = MSSeparator(
        model_type="bs_roformer",
        config_path="configs/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.yaml",
        model_path="pretrain/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        device='auto',
        device_ids=[0],
        output_format='mp3',
        use_tta=False,
        store_dirs={"vocals": "results/vocals", "instrumental": "results/instrumental"},
        audio_params = {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"},
        logger=get_logger(),
        debug=True
)

# Separate a folder of mixtures.
# Returns a list of successful separated files. Examples: ["input/test01.wav", "input/test02.wav"]
results_list = separator.process_folder("input")
separator.del_cache()

# Separate from a single numpy array.
# Returns a dict with separated audio arrays. Examples: {"vocals": np.ndarray, "instrumental": np.ndarray}
audio_array = separator.separate(mix: np.ndarray)
separator.del_cache()
```

### Init parameters

- `model_type: str`: The model type to use for inference. Choices: ['bs_roformer', 'mel_band_roformer', 'segm_models', 'htdemucs', 'mdx23c', 'swin_upernet', 'bandit', 'bandit_v2', 'scnet', 'scnet_unofficial', 'torchseg', 'apollo', 'bs_mamba2']
- `config_path: str`: The path to the configuration file for the model (*.yaml).
- `model_path: str`: The path to the model file (*.ckpt, *.th).
- `device: str`: The device to use for inference. Choices: ['auto', 'cpu', 'cuda', 'mps']. Set to 'auto' to automatically select the best device.
- `device_ids: List[int]`: The list of GPU IDs to use for inference. Only used when device is 'cuda'.
- `output_format: str`: The output format for separated files. Choices: ['wav', 'flac', 'mp3'].
- `use_tta: bool`: Whether to use test time augmentation for inference.
- `store_dirs: Union[str, Dict[str, Union[str, List[str]]]]`: The folder to store separated files.

  - `str` or `dict` can be used for output_folder. If str is used, all separated stems will be stored in the same folder. Exammple:
  ```python
  store_dirs="output_dir"
  ```

  - If dict is used, separated stems will be stored in different folders based on the keys. The keys in the dict are the names of the stems, and the values are the paths to the folders where the stems will be stored. Refer to the model configuration file (*.yaml) for the names of the stems. The stems are under `config.training.instruments`. If you can not find the configuration file, see [data_backup/msst_model_map.json](../data_backup/msst_model_map.json), which contains the mapping between the model name and the configuration file. Example:
  ```python
  store_dirs={
    "vocals": "results/vocals", 
    "instrumental": "results/instrumental"
  }
  ```

  - Also, the value can be list, which means the stem will be stored in multiple folders. Example:
  ```python
  store_dirs={
    "vocals": ["results/vocals", "results/vocals2"], 
    "instrumental": "results/instrumental"
  }
  ```

  - If you do not want to store a stem, you can omit it from the dict or use "" as its value. Example:
  ```python
  store_dirs={"vocals": "results/vocals", "instrumental": ""}
  # or, just omit the key:
  store_dirs={"vocals": "results/vocals"}
  ```

  - `audio_params: Dict[str, str]`: The parameters for audio encoding.
  - `wav_bit_depth: str`: The bit depth for WAV files. Choices: ['PCM_16', 'PCM_24', 'PCM_32', 'FLOAT'].
  - `flac_bit_depth: str`: The bit depth for FLAC files. Choices: ['PCM_16', 'PCM_24'].
  - `mp3_bit_rate: str`: The bit rate for MP3 files. Choices: ['96k', '128k', '192k', '256k', '320k'].

- `logger: logging.Logger`: The logger to use for logging. Set to `None` to Automatically create a logger.
- `debug: bool`: Whether to enable debug logging.

### Functions

- `separator.process_folder(input_folder: str)`: Separate a folder of mixtures.
  - Inupts: `input_folder: str`: The folder with mixtures to process.
  - Returns: A list of successful separated files. Examples: ["input/test01.wav", "input/test02.wav"]

- `separator.separate(mix: np.ndarray)`: Separate from a single numpy array.
  - Inputs: `mix: np.ndarray`: The mixture to separate. The shape of the array should be (num_channels, num_samples). For example, a stereo mixture should have a shape of (2, num_samples).
  - Returns: A dict with separated audio arrays. Examples: {"vocals": np.ndarray, "instrumental": np.ndarray}. The keys are the names of the stems, and the values are the separated audio arrays. The shape of results are (num_samples, num_channels). For example, a stereo mixture results have a shape of (num_samples, 2).

- `separator.save_audio(audio: np.ndarray, sr: int, file_name: str, store_dir: str)`: Save audio to a file.
  - Inputs: `audio: np.ndarray`: The audio to save. The shape of the array should be (num_samples, num_channels). For example, a stereo audio should have a shape of (num_samples, 2).
  - Inputs: `sr: int`: The sample rate of the audio.
  - Inputs: `file_name: str`: The name of the file to save, not including the extension. For example, "result01".
  - Inputs: `store_dir: str`: The directory to save the file.
  - Returns: None

- `separator.del_cache()`: Delete the cache files. Must be called after the inference is done.

## VR API

Here is a simple class calling method.

```python
from inference.vr_infer import VRSeparator
from utils.logger import get_logger

separator = VRSeparator(
    logger=get_logger(),
    debug=True,
    model_file="pretrain/VR_Models/1_HP-UVR.pth",
    output_dir={"Vocals": "results/Vocals", "Instrumental": "results/instrumental"},
    output_format="mp3",
    use_cpu=False,
    vr_params={"batch_size": 2, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
    audio_params = {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"},
)

# Separate a folder of mixtures.
# Returns a list of successful separated files. Examples: ["input/test01.wav", "input/test02.wav"]
results_list = separator.process_folder("input")
separator.del_cache()

# Separate from a single numpy array.
# Returns a dict with separated audio arrays. Examples: {"Vocals": np.ndarray, "Instrumental": np.ndarray}
audio_array = separator.separate(mix: np.ndarray)
separator.del_cache()
```

### Init parameters

- `logger: logging.Logger`: The logger to use for logging. Set to `None` to Automatically create a logger.
- `debug: bool`: Whether to enable debug logging.
- `model_file: str`: The path to the model file (*.pth). The model file should be a PyTorch model file.
- `output_dir: Union[str, Dict[str, Union[str, List[str]]]]`: The folder to store separated files.

  - `str` or `dict` can be used for output_folder. If str is used, all separated stems will be stored in the same folder. Exammple:
  ```python
  output_dir="output_dir"
  ```

  - If dict is used, separated stems will be stored in different folders based on the keys. The keys in the dict are the names of primary stem and secondary stems, and the values are the paths to the folders where the stems will be stored. Refer to [data_backup/vr_model_map.json](../data_backup/vr_model_map.json), which contains the mapping between the name of the model and it's primary stem name and secondary stem name. Example:
  ```python
  output_dir={
    "Vocals": "results/vocals", 
    "Instrumental": "results/instrumental"
  }
  ```

  - Also, the value can be list, which means the stem will be stored in multiple folders. Example:
  ```python
  output_dir={
    "Vocals": ["results/vocals", "results/vocals2"], 
    "Instrumental": "results/instrumental"
  }
  ```

  - If you do not want to store a stem, you can omit it from the dict or use "" as its value. Example:
  ```python
  output_dir={"Vocals": "results/vocals", "Instrumental": ""}
  # or, just omit the key:
  output_dir={"Vocals": "results/vocals"}
  ```

- `output_format: str`: The output format for separated files. Choices: ['wav', 'flac', 'mp3'].
- `use_cpu: bool`: Focusing CPU for inference.
- `vr_params: Dict[str, Any]`: The parameters for the VR model.
  - `batch_size: int`: Number of batches to process at a time. higher = more RAM, slightly faster processing.
  - `window_size: int`: Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality.
  - `aggression: int`: Intensity of primary stem extraction, -100 - 100. typically 5 for vocals & instrumentals.
  - `enable_tta: bool`: Enable Test-Time-Augmentation, slow but improves quality.
  - `enable_post_process: bool`: Identify leftover artifacts within vocal output, may improve separation for some songs.
  - `post_process_threshol: float`: Threshold for post_process feature: 0.1-0.3.
  - `high_end_process: bool`: Mirror the missing frequency range of the output.

- `audio_params: Dict[str, str]`: The parameters for audio encoding.
  - `wav_bit_depth: str`: The bit depth for WAV files. Choices: ['PCM_16', 'PCM_24', 'PCM_32', 'FLOAT'].
  - `flac_bit_depth: str`: The bit depth for FLAC files. Choices: ['PCM_16', 'PCM_24'].
  - `mp3_bit_rate: str`: The bit rate for MP3 files. Choices: ['96k', '128k', '192k', '256k', '320k'].

### Functions

- `separator.process_folder(input_folder: str)`: Separate a folder of mixtures.
  - Inupts: `input_folder: str`: The folder with mixtures to process.
  - Returns: A list of successful separated files. Examples: ["input/test01.wav", "input/test02.wav"]

- `separator.separate(mix: Union[np.ndarray, str])`: Separate from a single numpy array or audio file.
  - Inputs: `mix: Union[np.ndarray, str]`: The mixture to separate. If mix is np.ndarray, The shape of the array should be (num_channels, num_samples). For example, a stereo mixture should have a shape of (2, num_samples). If mix is str, it should be the path to the audio file.
  - Returns: A dict with separated audio arrays. Examples: {"vocals": np.ndarray, "instrumental": np.ndarray}. The keys are the names of the stems, and the values are the separated audio arrays. The shape of results are (num_samples, num_channels). For example, a stereo mixture results have a shape of (num_samples, 2).

- `separator.save_audio(audio: np.ndarray, sr: int, file_name: str, store_dir: str)`: Save audio to a file.
  - Inputs: `audio: np.ndarray`: The audio to save. The shape of the array should be (num_samples, num_channels). For example, a stereo audio should have a shape of (num_samples, 2).
  - Inputs: `sr: int`: The sample rate of the audio.
  - Inputs: `file_name: str`: The name of the file to save, not including the extension. For example, "result01".
  - Inputs: `store_dir: str`: The directory to save the file.
  - Returns: None

- `separator.del_cache()`: Delete the cache files. Must be called after the inference is done.

## msst_cli.py

Use `scripts\msst_cli.py`. When using cli, the `output_folder` can only be str.

```bash
usage: msst_cli.py [-h] [-d] [--device {auto,cpu,cuda,mps}] [--device_ids DEVICE_IDS [DEVICE_IDS ...]] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER] [--output_format {wav,flac,mp3}] --model_type MODEL_TYPE --model_path
                   MODEL_PATH --config_path CONFIG_PATH [--use_tta] [--wav_bit_depth {PCM_16,PCM_24,PCM_32,FLOAT}] [--flac_bit_depth {PCM_16,PCM_24}] [--mp3_bit_rate {96k,128k,192k,256k,320k}]

Music Source Separation Command Line Interface

options:
  -h, --help                                       show this help message and exit
  -d, --debug                                      Enable debug logging (default: False). Example: --debug
  --device {auto,cpu,cuda,mps}                     Device to use for inference (default: auto). Example: --device=cuda
  --device_ids DEVICE_IDS [DEVICE_IDS ...]         List of gpu ids, only used when device is cuda (default: 0). Example: --device_ids 0 1

Separation I/O Params:
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER     Folder with mixtures to process. (default: input). Example: --input_folder=input
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER  Folder to store separated files. Only can be str when using cli (default: results). Example: --output_folder=results
  --output_format {wav,flac,mp3}                   Output format for separated files (default: wav). Example: --output_format=wav

Model Params:
  --model_type MODEL_TYPE                          One of ['bs_roformer', 'mel_band_roformer', 'segm_models', 'htdemucs', 'mdx23c', 'swin_upernet', 'bandit', 'bandit_v2', 'scnet', 'scnet_unofficial', 'torchseg', 'apollo', 'bs_mamba2'].
  --model_path MODEL_PATH                          Path to model checkpoint.
  --config_path CONFIG_PATH                        Path to config file.
  --use_tta                                        Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality (default: False). Example: --use_tta

Audio Params:
  --wav_bit_depth {PCM_16,PCM_24,PCM_32,FLOAT}     Bit depth for wav output (default: FLOAT). Example: --wav_bit_depth=PCM_32
  --flac_bit_depth {PCM_16,PCM_24}                 Bit depth for flac output (default: PCM_24). Example: --flac_bit_depth=PCM_24
  --mp3_bit_rate {96k,128k,192k,256k,320k}         Bit rate for mp3 output (default: 320k). Example: --mp3_bit_rate=320k
```

For example:

```bash
python scripts/msst_cli.py -i input -o results --model_type bs_roformer --model_path models/bs_roformer.ckpt --config_path configs/bs_roformer.yaml --use_tta --output_format wav --wav_bit_depth PCM_32
```

`wav_bit_depth` only works when `output_format` is `wav`. `flac_bit_depth` only works when `output_format` is `flac`. `mp3_bit_rate` only works when `output_format` is `mp3`.

## vr_cli.py

Use `scripts\vr_cli.py`. When using cli, the `output_folder` can only be str.

```bash
usage: vr_cli.py [-h] [-d] [--use_cpu] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER] [--output_format {wav,flac,mp3}] -m MODEL_PATH [--batch_size BATCH_SIZE] [--window_size WINDOW_SIZE]
                 [--aggression AGGRESSION] [--enable_tta] [--high_end_process] [--enable_post_process] [--post_process_threshold POST_PROCESS_THRESHOLD] [--wav_bit_depth {PCM_16,PCM_24,PCM_32,FLOAT}]
                 [--flac_bit_depth {PCM_16,PCM_24}] [--mp3_bit_rate {96k,128k,192k,256k,320k}]

Vocal Remover Command Line Interface

options:
  -h, --help                                       show this help message and exit
  -d, --debug                                      Enable debug logging (default: False). Example: --debug
  --use_cpu                                        Use CPU instead of GPU for inference (default: False). Example: --use_cpu

Separation I/O Params:
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER     Folder with mixtures to process.
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER  Folder to store separated files. Only can be str when using cli (default: results). Example: --output_folder=results
  --output_format {wav,flac,mp3}                   Output format for separated files (default: wav). Example: --output_format=wav

VR Architecture Parameters:
  -m MODEL_PATH, --model_path MODEL_PATH           Path to model checkpoint.
  --batch_size BATCH_SIZE                          Number of batches to process at a time. higher = more RAM, slightly faster processing (default: 2). Example: --batch_size=16
  --window_size WINDOW_SIZE                        Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: 512). Example: --window_size=320
  --aggression AGGRESSION                          Intensity of primary stem extraction, -100 - 100. typically 5 for vocals & instrumentals (default: 5). Example: --aggression=2
  --enable_tta                                     Enable Test-Time-Augmentation, slow but improves quality (default: False). Example: --enable_tta
  --high_end_process                               Mirror the missing frequency range of the output (default: False). Example: --high_end_process
  --enable_post_process                            Identify leftover artifacts within vocal output, may improve separation for some songs (default: False). Example: --enable_post_process
  --post_process_threshold POST_PROCESS_THRESHOLD  Threshold for post_process feature: 0.1-0.3 (default: 0.2). Example: --post_process_threshold=0.1

Audio Params:
  --wav_bit_depth {PCM_16,PCM_24,PCM_32,FLOAT}     Bit depth for wav output (default: FLOAT). Example: --wav_bit_depth=PCM_32
  --flac_bit_depth {PCM_16,PCM_24}                 Bit depth for flac output (default: PCM_24). Example: --flac_bit_depth=PCM_24
  --mp3_bit_rate {96k,128k,192k,256k,320k}         Bit rate for mp3 output (default: 320k). Example: --mp3_bit_rate=320k
```

For example:

```bash
python scripts/vr_cli.py -i input_folder -o output_folder -m my_vr_model.pth --batch_size 4 --window_size 1024 --aggression 5 --enable_tta --post_process_threshold 0.2 --output_format flac --flac_bit_depth PCM_16
```

`wav_bit_depth` only works when `output_format` is `wav`. `flac_bit_depth` only works when `output_format` is `flac`. `mp3_bit_rate` only works when `output_format` is `mp3`.

## preset_infer_cli.py

Use `scripts\preset_infer_cli.py`.

```bash
usage: preset_infer_cli.py [-h] -p PRESET_PATH [-i INPUT_DIR] [-o OUTPUT_DIR] [-f {wav,mp3,flac}] [--debug] [--extra_output_dir]

Preset inference Command Line Interface

options:
  -h, --help                                         show this help message and exit
  -p PRESET_PATH, --preset_path PRESET_PATH          Path to the preset file (*.json). To create a preset file, please refer to the documentation or use WebUI to create one.
  -i INPUT_DIR, --input_dir INPUT_DIR                Path to the input folder
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR             Path to the output folder
  -f {wav,mp3,flac}, --output_format {wav,mp3,flac}  Output format of the audio
  --debug                                            Enable debug mode
  --extra_output_dir                                 Enable extra output directory
```

For example:

```bash
python scripts/preset_infer_cli.py -p preset.json -i input_folder -o output_folder -f wav --extra_output_dir
```

- The audio parameters such as `wav_bit_depth`, `flac_bit_depth`, and `mp3_bit_rate` will use the values in `data/webui_config.json`. If none are available, the default values will be used.
- The `--extra_output_dir` option will create a folder named `extra_output` in the output folder. All stems specified in `output_to_storage` in the preset file will be stored in this folder. If `--extra_output_dir` is not specified, the stems will be directly stored in the output folder.

### Preset format

We use json to store the preset file. You can create a preset file manually or use WebUI to create one. The preset file should contain the following fields:
- `version`: the version of the preset file, currently we only support `1.0.0`.
- `name`: the name of the preset file. It's optional.
- `flow`: a list of models to be used in the inference process. Each model should contain the following fields:
  - `model_type`: the type of the model. There are four types: `vocal_models`, `multi_stem_models`, `single_stem_models`, `UVR_VR_Models`.
  - `model_name`: the name of the model. It's the name of the model file in the `pretrain/{model_type}` folder.
  - `input_to_next`: the input of the next model. The name is the stem name output from this step's inference.
  - `output_to_storage`: list of stems to be directly stored in the output folder. They will not be used as input for the next model. The names are the stem name output from this step's inference. If none are specified (`"output_to_storage": []`), no stems will be stored in the output folder at this step.

The following json is an example of a preset file:
```json
{
    "version": "1.0.0",
    "name": "example",
    "flow": [
        {
            "model_type": "vocal_models",
            "model_name": "kimmel_unwa_ft.ckpt",
            "input_to_next": "vocals",
            "output_to_storage": ["other"]
        },
        {
            "model_type": "UVR_VR_Models",
            "model_name": "5_HP-Karaoke-UVR.pth",
            "input_to_next": "Vocals",
            "output_to_storage": ["Instrumental", "Vocals"]
        },
        {
            "model_type": "vocal_models",
            "model_name": "dereverb_echo_mbr_fused_model.ckpt",
            "input_to_next": "dry",
            "output_to_storage": []
        }
    ]
}
```

## ensemble_cli.py

Use `scripts\ensemble_cli.py`. Input multiple raw audio files that have not been separated. The script first uses the preset JSON specified model for inference, and then ensembles the inference results.

```bash
usage: ensemble_cli.py [-h] [-p PRESET_PATH] [-m {avg_wave,median_wave,min_wave,max_wave,avg_fft,median_fft,min_fft,max_fft}] [-i INPUT_DIR] [-o OUTPUT_DIR] [-f {wav,mp3,flac}] [--extract_inst] [--debug]

Ensemble inference Command Line Interface

options:
  -h, --help                                                show this help message and exit
  -p PRESET_PATH, --preset_path PRESET_PATH                 Path to the ensemble preset data file (.json). To create a preset file, please refer to the documentation.
  -m {avg_wave,median_wave,min_wave,max_wave,avg_fft,median_fft,min_fft,max_fft}, --ensemble_mode {avg_wave,median_wave,min_wave,max_wave,avg_fft,median_fft,min_fft,max_fft}
                                                            Type of ensemble to perform.
  -i INPUT_DIR, --input_dir INPUT_DIR                       Path to the input folder
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR                    Path to the output folder
  -f {wav,mp3,flac}, --output_format {wav,mp3,flac}         Output format of the audio
  --extract_inst                                            Extract instruments by subtracting ensemble result from raw audio
  --debug                                                   Enable debug mode
```

For example:

```bash
python scripts/ensemble_cli.py -p ensemble.json -i input -o output -f wav --extract_inst
```

- `ensemble_mode`: Type of ensemble to perform.
  - `avg_wave` - ensemble on 1D variant, find average for every sample of waveform independently
  - `median_wave` - ensemble on 1D variant, find median value for every sample of waveform independently
  - `min_wave` - ensemble on 1D variant, find minimum absolute value for every sample of waveform independently
  - `max_wave` - ensemble on 1D variant, find maximum absolute value for every sample of waveform independently
  - `avg_fft` - ensemble on spectrogram (Short-time Fourier transform (STFT), 2D variant), find average for every pixel of spectrogram independently. After averaging use inverse STFT to obtain original 1D-waveform back.
  - `median_fft` - the same as avg_fft but use median instead of mean (only useful for ensembling of 3 or more sources).
  - `min_fft` - the same as avg_fft but use minimum function instead of mean (reduce aggressiveness).
  - `max_fft` - the same as avg_fft but use maximum function instead of mean (the most aggressive).

  - Notes
    - `min_fft` can be used to do more conservative ensemble - it will reduce influence of more aggressive models.
    - It's better to ensemble models which are of equal quality - in this case it will give gain. If one of model is bad - it will reduce overall quality.
    - In ZFTurbo's experiments `avg_wave` was always better or equal in SDR score comparing with other methods.

- `extract_inst`：Extract instruments by subtracting ensemble result from raw audio.
- The audio parameters such as `wav_bit_depth`, `flac_bit_depth`, and `mp3_bit_rate` will use the values in `data/webui_config.json`. If none are available, the default values will be used.

### Ensemble preset format

We use json to store the preset file. Ensemble preset can not be made in WebUI, you mush create it manually. The preset file should have the following format:
- `flow`: list of models to be used for inference. Each model is represented as a dictionary with the following keys:
- `model_type`: type of the model. There are four types: `vocal_models`, `multi_stem_models`, `single_stem_models`, `UVR_VR_Models`.
- `model_name`: name of the model. It's the name of the model file in the `pretrain/{model_type}` folder.
- `stem`: stem to ensemble. Each model outputs 2 or more stems but can only ensemble only one stem. It needs to be specified in the preset file.
- `weight`: ensemble weight (float). It's the weight of the model in the ensemble. The default weight is 1.

The following json is an example of a ensemble preset file:
```json
{
    "flow": [
        {
            "model_type": "vocal_models",
            "model_name": "kimmel_unwa_ft.ckpt",
            "stem": "other",
            "weight": 1
        },
        {
            "model_type": "vocal_models",
            "model_name": "inst_v1e.ckpt",
            "stem": "other",
            "weight": 1
        },
        {
            "model_type": "vocal_models",
            "model_name": "melband_roformer_inst_v2.ckpt",
            "stem": "other",
            "weight": 1
        }
    ]
}
```

## ensemble_audio_cli.py

Use `scripts\ensemble_audio_cli.py`. Input multiple audio of the same duration for ensemble. It can be used to ensemble results of different algorithms.

```bash
usage: ensemble_audio_cli.py [-h] --files FILES [FILES ...] [--type {avg_wave,median_wave,min_wave,max_wave,avg_fft,median_fft,min_fft,max_fft}] [--weights WEIGHTS [WEIGHTS ...]] [--output_dir OUTPUT_DIR]

Ensemble from audio files

options:
  -h, --help                                                show this help message and exit
  --files FILES [FILES ...]                                 Path to all audio-files to ensemble
  --type {avg_wave,median_wave,min_wave,max_wave,avg_fft,median_fft,min_fft,max_fft}
                                                            Type of ensemble to perform.
  --weights WEIGHTS [WEIGHTS ...]                           Weights to create ensemble. Number of weights must be equal to number of files
  --output_dir OUTPUT_DIR                                   Path to wav file where ensemble result will be stored
```

For example:

```bash
python scripts/ensemble_audio_cli.py --files "audio1.wav" "audio2.wav" --type avg_fft --weights 1 1 --output_dir "ensemble.wav"
```

- `type`: Type of ensemble to perform.
  - `avg_wave` - ensemble on 1D variant, find average for every sample of waveform independently
  - `median_wave` - ensemble on 1D variant, find median value for every sample of waveform independently
  - `min_wave` - ensemble on 1D variant, find minimum absolute value for every sample of waveform independently
  - `max_wave` - ensemble on 1D variant, find maximum absolute value for every sample of waveform independently
  - `avg_fft` - ensemble on spectrogram (Short-time Fourier transform (STFT), 2D variant), find average for every pixel of spectrogram independently. After averaging use inverse STFT to obtain original 1D-waveform back.
  - `median_fft` - the same as avg_fft but use median instead of mean (only useful for ensembling of 3 or more sources).
  - `min_fft` - the same as avg_fft but use minimum function instead of mean (reduce aggressiveness).
  - `max_fft` - the same as avg_fft but use maximum function instead of mean (the most aggressive).

- Notes
  - `min_fft` can be used to do more conservative ensemble - it will reduce influence of more aggressive models.
  - It's better to ensemble models which are of equal quality - in this case it will give gain. If one of model is bad - it will reduce overall quality.
  - In ZFTurbo's experiments `avg_wave` was always better or equal in SDR score comparing with other methods.

## some_cli.py

Use `scripts\some_cli.py`. SOME is a tool to extract the midi file from the audio file. It only supports for Vocal.

```bash
usage: some_cli.py [-h] [-m MODEL] [-c CONFIG] -i INPUT_AUDIO [-o OUTPUT_DIR] [-t TEMPO]

SOME Command Line Interface

options:
  -h, --help                                 show this help message and exit
  -m MODEL, --model MODEL                    Path to the model checkpoint (*.ckpt)
  -c CONFIG, --config CONFIG                 Path to the config file (*.yaml)
  -i INPUT_AUDIO, --input_audio INPUT_AUDIO  Path to the input audio file
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR     Path to the output folder
  -t TEMPO, --tempo TEMPO                    Specify tempo in the output MIDI
```

For example:

```bash
python scripts/some_cli.py -m "tools/SOME_weights/model_steps_64000_simplified.ckpt" -c "configs_backup/config_some.yaml" -i "audio.wav" -o "output" -t 120
```

Notes
- You can download some model from [here](https://huggingface.co/Sucial/MSST-WebUI/resolve/main/SOME_weights/model_steps_64000_simplified.ckpt). The configuration file of the model **must use the revised configuration file we provide**, which is located at `configs_backup/config_some.yaml`.
- Audio BPM (beats per minute) can be measured using software like MixMeister BPM Analyzer.
- To ensure MIDI extraction quality, please use clean, clear, and noise-free vocal audio without reverb.
- The output MIDI does not contain lyrics information; you need to add the lyrics manually.
- In actual use, some notes may appear disconnected, requiring manual correction. The SOME model is mainly designed for auto-labeling with DiffSinger vocal models, which may lead to finer segmentation of notes than typically needed in user creations.
- The extracted MIDI is not quantized/aligned with the beat/does not match the BPM, requiring manual adjustment in editors.
