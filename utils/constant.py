"""
Configuration File for Model Management

This script defines various paths and constants used throughout the application.
It includes configurations for web UI settings, model data, language files,
backup directories, and executable paths for required tools like FFmpeg and Python.

Key Configurations:
- WEBUI_CONFIG: Path to the main web UI configuration file.
- WEBUI_CONFIG_BACKUP: Backup path for the web UI configuration.
- PRESETS: Path for preset data storage.
- MSST_MODEL: Path for the MSST model mapping. (Removed in >1.7)
- VR_MODEL: Path for the VR model mapping. (Removed in >1.7)
- LANGUAGE: Path for language data files.
- BACKUP: Directory for storing backups.
- MODEL_FOLDER: Directory for pre-trained models.
- TEMP_PATH: Temporary directory for processing.
- UNOFFICIAL_MODEL: Path for unofficial model configurations.
- VR_MODELPARAMS: Directory for VR model parameters.
- UPDATE_URL: URL for checking updates.
- SOME_WEIGHT: Path to SOME weight file.
- THEME_FOLDER: Path to the theme folder.
- METRICS: List of available metrics for model evaluation.

Model Types and Choices:
- MODEL_TYPE: List of available MSST model types.
- MODEL_CHOICES: List of unofficial model categories.

Executable Paths:
- FFMPEG: Path to the FFmpeg executable.
- PYTHON: Path to the Python executable.

Usage:
Import this configuration file to access defined paths and settings throughout the application.
"""

import os
import sys
import json

# webui config path
WEBUI_CONFIG = "data/webui_config.json"
WEBUI_CONFIG_BACKUP = "data_backup/webui_config.json"

# package version
with open(WEBUI_CONFIG_BACKUP, "r") as f:
	config = json.load(f)
	PACKAGE_VERSION = config.get("version", "Unknown version")

# presets data path
PRESETS = "presets"
PRESETS_BACKUP = "presets_backup"

# preset version
PRESET_VERSION = "1.0.0"
SUPPORTED_PRESET_VERSION = ["1.0.0"]

# # msst model map path
# MSST_MODEL = "data/msst_model_map.json"
# MSST_MODEL_BACKUP = "data_backup/msst_model_map.json"

# vr model map path
# VR_MODEL = "data/vr_model_map.json"
# VR_MODEL_BACKUP = "data_backup/vr_model_map.json"

# language data path
LANGUAGE = "data/language.json"
LANGUAGE_BACKUP = "data_backup/language.json"

# path to models information config
MODELS_INFO = "data/models_info.json"

# path to pretrain folder
MODEL_FOLDER = "pretrain"

# path to temp folder
TEMP_PATH = "tmpdir"

# path to unoffical model config
# path to unofficial msst model config: UNOFFICIAL_MODEL/unofficial_msst_model.json
# path to unofficial vr model config: UNOFFICIAL_MODEL/unofficial_vr_model.json
UNOFFICIAL_MODEL = "config_unofficial"

# path to vr model params
# path to unofficial vr model params: UNOFFICIAL_MODEL/vr_modelparams
VR_MODELPARAMS = "configs/vr_modelparams"

# url for check for updates
UPDATE_URL = "https://github.com/SUC-DriverOld/MSST-WebUI/releases/latest"

# pretrained SOME weight
SOME_WEIGHT = "tools/SOME_weights/model_steps_64000_simplified.ckpt"
SOME_CONFIG = "configs_backup/config_some.yaml"

# path to the theme folder
THEME_FOLDER = "tools/themes"

# msst model types, type=list
MODEL_TYPE = ["bs_roformer", "mel_band_roformer", "segm_models", "htdemucs", "mdx23c", "swin_upernet", "bandit", "bandit_v2", "scnet", "scnet_unofficial", "torchseg", "apollo", "bs_mamba2"]

# model choices (unofficial), type=list
MODEL_CHOICES = ["vocal_models", "multi_stem_models", "single_stem_models", "UVR_VR_Models"]

# metrics for model evaluation and training
METRICS = ["sdr", "l1_freq", "si_sdr", "log_wmse", "aura_stft", "aura_mrstft", "bleedless", "fullness"]

# ensemble modes
ENSEMBLE_MODES = ["avg_wave", "median_wave", "min_wave", "max_wave", "avg_fft", "median_fft", "min_fft", "max_fft"]

# ffmpeg executable path, if not found, use system ffmpeg
FFMPEG = ".\\ffmpeg\\bin\\ffmpeg.exe" if os.path.isfile(".\\ffmpeg\\bin\\ffmpeg.exe") else "ffmpeg"

# python executable path, if not found, use current python
PYTHON = ".\\workenv\\python.exe" if os.path.isfile(".\\workenv\\python.exe") else sys.executable
