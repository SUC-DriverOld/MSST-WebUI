## Training & Validation

- Refer to [dataset_types.md](dataset_types.md) for details about dataset types. 
- Refer to [arugments.md](arguments.md) for details about arguments. 
- Refer to [bs_roformer_info.md](bs_roformer_info.md) for details about `bs_roformer` model.
- Refer to [mel_roformer_experiments.md](mel_roformer_experiments.md) for details about pretrained `mel_band_roformer` models.
- Refer ro [pretrained_models.md](pretrained_models.md) for details about all pretrained models.

### Training

First, you need to prepare dataset. You can use [dataset_types.md](dataset_types.md) to prepare dataset.<br>
Use `train/train.py`. If you use multi-GPUs, try to use `train/train_accelerate.py`.

```bash
usage: train.py [-h] [--model_type MODEL_TYPE] [--config_path CONFIG_PATH] [--start_check_point START_CHECK_POINT] [--results_path RESULTS_PATH] [--data_path DATA_PATH [DATA_PATH ...]]
                [--dataset_type DATASET_TYPE] [--valid_path VALID_PATH [VALID_PATH ...]] [--num_workers NUM_WORKERS] [--pin_memory PIN_MEMORY] [--seed SEED]
                [--device_ids DEVICE_IDS [DEVICE_IDS ...]] [--use_multistft_loss] [--use_mse_loss] [--use_l1_loss] [--pre_valid]
                [--metrics {sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness} [{sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness} ...]]
                [--metric_for_scheduler {sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness}]

options:
  -h, --help                                                show this help message and exit
  --model_type MODEL_TYPE                                   One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit
  --config_path CONFIG_PATH                                 path to config file
  --start_check_point START_CHECK_POINT                     Initial checkpoint to start training
  --results_path RESULTS_PATH                               path to folder where results will be stored (weights, metadata)
  --data_path DATA_PATH [DATA_PATH ...]                     Dataset data paths. You can provide several folders.
  --dataset_type DATASET_TYPE                               Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md
  --valid_path VALID_PATH [VALID_PATH ...]                  validation data paths. You can provide several folders.
  --num_workers NUM_WORKERS                                 dataloader num_workers
  --pin_memory PIN_MEMORY                                   dataloader pin_memory
  --seed SEED                                               random seed
  --device_ids DEVICE_IDS [DEVICE_IDS ...]                  list of gpu ids
  --use_multistft_loss                                      Use MultiSTFT Loss (from auraloss package)
  --use_mse_loss                                            Use default MSE loss
  --use_l1_loss                                             Use L1 loss
  --pre_valid                                               Run validation before training
  --metrics {sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness} [{sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness} ...]
                                                            List of metrics to use.
  --metric_for_scheduler {sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness}
                                                            Metric which will be used for scheduler.
```

### Validation

Use `train/valid.py`

```bash
usage: valid.py [-h] [--model_type MODEL_TYPE] [--config_path CONFIG_PATH] [--start_check_point START_CHECK_POINT] [--valid_path VALID_PATH [VALID_PATH ...]] [--store_dir STORE_DIR]
                [--device_ids DEVICE_IDS [DEVICE_IDS ...]] [--num_workers NUM_WORKERS] [--pin_memory PIN_MEMORY] [--extension EXTENSION] [--use_tta]
                [--metrics {sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness} [{sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness} ...]]

options:
  -h, --help                                                show this help message and exit
  --model_type MODEL_TYPE                                   One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit
  --config_path CONFIG_PATH                                 path to config file
  --start_check_point START_CHECK_POINT                     Initial checkpoint to valid weights
  --valid_path VALID_PATH [VALID_PATH ...]                  validate path
  --store_dir STORE_DIR                                     path to store results as wav file
  --device_ids DEVICE_IDS [DEVICE_IDS ...]                  list of gpu ids
  --num_workers NUM_WORKERS                                 dataloader num_workers
  --pin_memory PIN_MEMORY                                   dataloader pin_memory
  --extension EXTENSION                                     Choose extension for validation
  --use_tta                                                 Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality.
  --metrics {sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness} [{sdr,l1_freq,si_sdr,log_wmse,aura_stft,aura_mrstft,bleedless,fullness} ...]
                                                            List of metrics to use.
```