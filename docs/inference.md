# Inference Usage

## 1. Music Source Separation Inference

### Command Line Interface (CLI)

Use `inference/msst_cli.py`.

```bash
usage: msst_cli.py [-h] [-d] [--device {auto,cpu,cuda,mps}] [--device_ids DEVICE_IDS [DEVICE_IDS ...]] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER] [--output_format {wav,flac,mp3}]
                   [--model_type MODEL_TYPE] [--model_path MODEL_PATH] [--config_path CONFIG_PATH] [--use_tta]

Music Source Separation Command Line Interface

options:
  -h, --help                                       show this help message and exit
  -d, --debug                                      Enable debug logging (default: False). Example: --debug
  --device {auto,cpu,cuda,mps}                     Device to use for inference (default: auto). Example: --device=cuda
  --device_ids DEVICE_IDS [DEVICE_IDS ...]         List of gpu ids, only used when device is cuda (default: 0). Example: --device_ids 0 1

Separation I/O Params:
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER     Folder with mixtures to process. [required]
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER  Folder to store separated files. str for single folder, dict with instrument keys for multiple folders. Example: --output_folder=results or --output_folder="{'vocals': 'results/vocals', 'instrumental': 'results/instrumental'}"
  --output_format {wav,flac,mp3}                   Output format for separated files (default: wav). Example: --output_format=wav

Model Params:
  --model_type MODEL_TYPE                          One of ['bs_roformer', 'mel_band_roformer', 'segm_models', 'htdemucs', 'mdx23c', 'swin_upernet', 'bandit', 'bandit_v2', 'scnet', 'scnet_unofficial', 'torchseg', 'apollo', 'bs_mamba2']. [required]
  --model_path MODEL_PATH                          Path to model checkpoint. [required]
  --config_path CONFIG_PATH                        Path to config file. [required]
  --use_tta                                        Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality (default: False). Example: --use_tta
```

### Python API

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
        store_dirs={
            "vocals": "results/vocals",
            "instrumental": "results/instrumental",
            },
        logger=get_logger(),
        debug=True
)

# Separate a folder of mixtures.
# Returns a list of successful separated files. Examples: ["input/test01.wav", "input/test02.wav"]
results_list = separator.process_folder("input")
separator.del_cache()

# Separate from a single numpy array
# Returns a dict with separated audio arrays. Examples: {"vocals": NDArray, "instrumental": NDArray}
import numpy as np
mix = np.load("input/test02.npy") # load numpy array for example
audio_array = separator.separate(mix)
separator.del_cache()
```

> [!NOTE]
> ### 1. About `store_dirs`
> 
> `str` or `dict` can be used for output_folder. If str is used, all separated stems will be stored in the same folder. If dict is used, separated stems will be stored in different folders based on the keys. Example:
> ```python
> store_dirs={
>    "vocals": "results/vocals", 
>    "instrumental": "results/instrumental"
> }
> ```
> Also, the value can be list, which means the stem will be stored in multiple folders. Example:
> ```python
> store_dirs={
>    "vocals": ["results/vocals", "results/vocals2"], 
>    "instrumental": "results/instrumental"
> }
> ```
>
> The keys in the dict are the names of the stems, and the values are the paths to the folders where the stems will be stored. **If you do not want to store a stem, you can omit it from the dict or use "" as its value.**
> 
> ### 2. NDarray as input
> 
> You don't need to transpose the numpy array. For example: audio.shape=(2, 838928) is fine.

## 2. Vocal Remover

### Command Line Interface (CLI)

Use `inference/vr_cli.py`

```bash
usage: vr_cli.py [-h] [-d] [--use_cpu] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER] [--output_format {wav,flac,mp3}] [-m MODEL_PATH] [--invert_spect] [--batch_size BATCH_SIZE] [--window_size WINDOW_SIZE]
                 [--aggression AGGRESSION] [--enable_tta] [--high_end_process] [--enable_post_process] [--post_process_threshold POST_PROCESS_THRESHOLD]

Vocal Remover Command Line Interface

options:
  -h, --help                                       show this help message and exit
  -d, --debug                                      Enable debug logging (default: False). Example: --debug
  --use_cpu                                        Use CPU instead of GPU for inference (default: False). Example: --use_cpu

Separation I/O Params:
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER     Folder with mixtures to process. [required]
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER  Folder to store separated files. str for single folder, dict with instrument keys for multiple folders. Example: --output_folder=results or --output_folder="{'vocals': 'results/vocals', 'instrumental': 'results/instrumental'}"
  --output_format {wav,flac,mp3}                   Output format for separated files (default: wav). Example: --output_format=wav

Common Separation Parameters:
  -m MODEL_PATH, --model_path MODEL_PATH           Path to model checkpoint. [required]
  --invert_spect                                   Invert secondary stem using spectogram (default: False). Example: --invert_spect

VR Architecture Parameters:
  --batch_size BATCH_SIZE                          Number of batches to process at a time. higher = more RAM, slightly faster processing (default: 2). Example: --batch_size=16
  --window_size WINDOW_SIZE                        Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality. (default: 512). Example: --window_size=320
  --aggression AGGRESSION                          Intensity of primary stem extraction, -100 - 100. typically 5 for vocals & instrumentals (default: 5). Example: --aggression=2
  --enable_tta                                     Enable Test-Time-Augmentation; slow but improves quality (default: False). Example: --enable_tta
  --high_end_process                               Mirror the missing frequency range of the output (default: False). Example: --high_end_process
  --enable_post_process                            Identify leftover artifacts within vocal output; may improve separation for some songs (default: False). Example: --enable_post_process
  --post_process_threshold POST_PROCESS_THRESHOLD  Threshold for post_process feature: 0.1-0.3 (default: 0.2). Example: --post_process_threshold=0.1
```

### Python API

```python
from modules.vocal_remover.separator import Separator
from utils.logger import get_logger

separator = Separator(
    logger=get_logger(),
    debug=True,
    model_file="pretrain/VR_Models/1_HP-UVR.pth",
    output_dir={
        "Vocals": "results/Vocals",
        "Instrumental": "results/instrumental",
        },
    output_format="mp3",
    invert_using_spec=False,
    use_cpu=False,
    vr_params={"batch_size": 2, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
)

# Separate a folder of mixtures.
# Returns a list of successful separated files. Examples: ["input/test01.wav", "input/test02.wav"]
results_list = separator.process_folder("input")
separator.del_cache()

# Separate from a single numpy array
# Returns a dict with separated audio arrays. Examples: {"vocals": NDArray, "instrumental": NDArray}
import numpy as np
mix = np.load("input/test02.npy") # load numpy array for example
audio_array = separator.separate(mix)
separator.del_cache()
```