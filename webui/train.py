__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import os.path
import gradio as gr
import multiprocessing
import subprocess

from utils.constant import *
from webui.utils import i18n, load_configs, save_configs, logger


def run_command(command):
	subprocess.run(command, shell=True)


def save_training_config(
	train_model_type,
	train_config_path,
	train_dataset_type,
	train_dataset_path,
	train_valid_path,
	train_num_workers,
	train_device_ids,
	train_seed,
	train_pin_memory,
	train_use_multistft_loss,
	train_use_mse_loss,
	train_use_l1_loss,
	train_results_path,
	train_accelerate,
	train_pre_validate,
	train_metrics_list,
	train_metrics_scheduler,
):
	config = load_configs(WEBUI_CONFIG)
	config["training"]["model_type"] = train_model_type
	config["training"]["config_path"] = train_config_path
	config["training"]["dataset_type"] = train_dataset_type
	config["training"]["dataset_path"] = train_dataset_path
	config["training"]["valid_path"] = train_valid_path
	config["training"]["num_workers"] = train_num_workers
	config["training"]["device"] = train_device_ids
	config["training"]["seed"] = train_seed
	config["training"]["pin_memory"] = train_pin_memory
	config["training"]["use_multistft_loss"] = train_use_multistft_loss
	config["training"]["use_mse_loss"] = train_use_mse_loss
	config["training"]["use_l1_loss"] = train_use_l1_loss
	config["training"]["accelerate"] = train_accelerate
	config["training"]["pre_valid"] = train_pre_validate
	config["training"]["results_path"] = train_results_path
	config["training"]["metrics"] = train_metrics_list
	config["training"]["metrics_scheduler"] = train_metrics_scheduler
	save_configs(config, WEBUI_CONFIG)
	logger.info(f"Saved training config: {config['training']}")
	return i18n("配置保存成功!")


def update_train_start_check_point(path):
	if not os.path.isdir(path):
		raise gr.Error(i18n("请先选择模型保存路径! "))

	ckpt_files = [f for f in os.listdir(path) if f.endswith((".ckpt", ".th"))]
	return gr.Dropdown(label=i18n("初始模型"), choices=ckpt_files if ckpt_files else ["None"])


def start_training(
	train_model_type,
	train_config_path,
	train_dataset_type,
	train_dataset_path,
	train_valid_path,
	train_num_workers,
	train_device_ids,
	train_seed,
	train_pin_memory,
	train_use_multistft_loss,
	train_use_mse_loss,
	train_use_l1_loss,
	train_results_path,
	train_start_check_point,
	train_accelerate,
	train_pre_validate,
	train_metrics_list,
	train_metrics_scheduler,
):
	if train_model_type not in MODEL_TYPE:
		return i18n("模型类型错误, 请重新选择")
	if not os.path.isfile(train_config_path):
		return i18n("配置文件不存在, 请重新选择")
	if not os.path.exists(train_dataset_path):
		return i18n("数据集路径不存在, 请重新选择")
	if not os.path.exists(train_valid_path):
		return i18n("验证集路径不存在, 请重新选择")
	if train_dataset_type not in [1, 2, 3, 4]:
		return i18n("数据集类型错误, 请重新选择")
	os.makedirs(train_results_path, exist_ok=True)

	gpu_ids = []
	if len(train_device_ids) == 0:
		return i18n("请选择GPU")
	for gpu in train_device_ids:
		gpu_ids.append(str(gpu[: gpu.index(":")]))

	device_ids = " ".join(gpu_ids)
	num_workers = int(train_num_workers)
	seed = int(train_seed)
	pin_memory = "--pin_memory" if train_pin_memory else ""
	use_multistft_loss = "--use_multistft_loss" if train_use_multistft_loss else ""
	use_mse_loss = "--use_mse_loss" if train_use_mse_loss else ""
	use_l1_loss = "--use_l1_loss" if train_use_l1_loss else ""
	pre_valid = "--pre_valid" if train_pre_validate else ""
	start_ckpt = os.path.join(train_results_path, train_start_check_point)
	start_check_point = f'--start_check_point "{start_ckpt}"' if os.path.isfile(start_ckpt) else ""
	metrics = " ".join(train_metrics_list)
	metrics_scheduler = train_metrics_scheduler

	if train_accelerate:
		logger.info("Accelerate training enabled")
		command = f'{PYTHON} train/train_accelerate.py --model_type {train_model_type} --config_path "{train_config_path}" {start_check_point} --results_path "{train_results_path}" --data_path "{train_dataset_path}" --dataset_type {train_dataset_type} --valid_path "{train_valid_path}" --num_workers {num_workers} --device_ids {device_ids} --seed {seed} {pin_memory} {use_multistft_loss} {use_mse_loss} {use_l1_loss} {pre_valid}'
	else:
		command = f'{PYTHON} train/train.py --model_type {train_model_type} --config_path "{train_config_path}" {start_check_point} --results_path "{train_results_path}" --data_path "{train_dataset_path}" --dataset_type {train_dataset_type} --valid_path "{train_valid_path}" --num_workers {num_workers} --device_ids {device_ids} --seed {seed} {pin_memory} {use_multistft_loss} {use_mse_loss} {use_l1_loss} {pre_valid} --metrics {metrics} --metric_for_scheduler {metrics_scheduler}'

	logger.info(f"Start training with command: {command}")
	msst_training = multiprocessing.Process(target=run_command, args=(command,), name="msst_training")
	msst_training.start()
	logger.debug(f"Training process started, PID: {msst_training.pid}")

	return i18n("训练启动成功! 请前往控制台查看训练信息! ")


def validate_model(
	valid_model_type, valid_config_path, valid_model_path, valid_path, valid_results_path, valid_device_ids, valid_num_workers, valid_extension, valid_pin_memory, valid_use_tta, vaild_metrics
):
	if valid_model_type not in MODEL_TYPE:
		return i18n("模型类型错误, 请重新选择")
	if not os.path.isfile(valid_config_path):
		return i18n("配置文件不存在, 请重新选择")
	if not os.path.isfile(valid_model_path):
		return i18n("模型不存在, 请重新选择")
	if not os.path.exists(valid_path):
		return i18n("验证集路径不存在, 请重新选择")
	os.makedirs(valid_results_path, exist_ok=True)

	gpu_ids = []
	if len(valid_device_ids) == 0:
		return i18n("请选择GPU")
	for gpu in valid_device_ids:
		gpu_ids.append(str(gpu[: gpu.index(":")]))

	device = " ".join(gpu_ids)
	pin_memory = "--pin_memory" if valid_pin_memory else ""
	use_tta = "--use_tta" if valid_use_tta else ""
	metrics = " ".join(vaild_metrics)

	command = f'{PYTHON} train/valid.py --model_type {valid_model_type} --config_path "{valid_config_path}" --start_check_point "{valid_model_path}" --valid_path "{valid_path}" --store_dir "{valid_results_path}" --device_ids {device} --num_workers {valid_num_workers} --extension {valid_extension} {pin_memory} {use_tta} --metrics {metrics}'

	logger.info(f"Start validation with command: {command}")
	msst_valid = multiprocessing.Process(target=run_command, args=(command,), name="msst_valid")
	msst_valid.start()
	logger.debug(f"Validation process started, PID: {msst_valid.pid}")
	msst_valid.join()

	return i18n("验证完成! 请打开输出文件夹查看详细结果")


def stop_msst_valid():
	for process in multiprocessing.active_children():
		if process.name == "msst_valid":
			process.terminate()
			process.join()
			logger.info(f"Inference process stopped, PID: {process.pid}")


def load_augmentations_config():
	try:
		with open("configs/augmentations_template.yaml", "r", encoding="utf-8") as f:
			return f.read()
	except FileNotFoundError:
		return i18n("错误: 无法找到增强配置文件模板, 请检查文件configs/augmentations_template.yaml是否存在。")
