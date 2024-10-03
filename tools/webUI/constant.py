"""
Configuration File for Model Management

This script defines various paths and constants used throughout the application.
It includes configurations for web UI settings, model data, language files,
backup directories, and executable paths for required tools like FFmpeg and Python.

Key Configurations:
- WEBUI_CONFIG: Path to the main web UI configuration file.
- WEBUI_CONFIG_BACKUP: Backup path for the web UI configuration.
- PRESETS: Path for preset data storage.
- MSST_MODEL: Path for the MSST model mapping.
- VR_MODEL: Path for the VR model mapping.
- LANGUAGE: Path for language data files.
- BACKUP: Directory for storing backups.
- MODEL_FOLDER: Directory for pre-trained models.
- TEMP_PATH: Temporary directory for processing.
- UNOFFICIAL_MODEL: Path for unofficial model configurations.
- VR_MODELPARAMS: Directory for VR model parameters.

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

# version
PACKAGE_VERSION = "1.6.2"

# webui config path, not the backup path
WEBUI_CONFIG = "data/webui_config.json"

# webui config backup path
WEBUI_CONFIG_BACKUP = "data_backup/webui_config.json"

# presets data path
PRESETS = "data/preset_data.json"

# msst model map path
MSST_MODEL = "data/msst_model_map.json"

# vr model map path
VR_MODEL = "data/vr_model_map.json"

# language data path
LANGUAGE = "data/language.json"

# path to backup folder
BACKUP = "backup"

# path to pretrain folder
MODEL_FOLDER = "pretrain"

# path to temp folder
TEMP_PATH = "tmpdir"

# path to unoffical model config
UNOFFICIAL_MODEL = "config_unofficial"

# path to vr model params
VR_MODELPARAMS = "configs/vr_modelparams"

# msst model types, type=list
MODEL_TYPE = [
    'bs_roformer', 
    'mel_band_roformer', 
    'segm_models', 
    'htdemucs', 
    'mdx23c', 
    'swin_upernet', 
    'bandit', 
    'bandit_v2', 
    'scnet', 
    'scnet_unofficial', 
    'torchseg', 
    'apollo', 
    'bs_mamba2'
    ]

# model choices (unofficial), type=list
MODEL_CHOICES = [
    "vocal_models", 
    "multi_stem_models", 
    "single_stem_models", 
    "UVR_VR_Models"
    ]

# ffmpeg executable path, if not found, use system ffmpeg
FFMPEG = ".\\ffmpeg\\bin\\ffmpeg.exe" if os.path.isfile(".\\ffmpeg\\bin\\ffmpeg.exe") else "ffmpeg"

# python executable path, if not found, use current python
PYTHON = ".\\workenv\\python.exe" if os.path.isfile(".\\workenv\\python.exe") else sys.executable

# url for check for updates
UPDATE_URL = "https://github.com/SUC-DriverOld/MSST-WebUI/releases/latest"

# pretrained SOME weight
SOME_WEIGHT = "tools/SOME_weights/model_steps_64000_simplified.ckpt"