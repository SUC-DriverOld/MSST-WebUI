import yaml
import os
import json
from ml_collections import ConfigDict
import shutil


def clear_folder(folder_path):
	if not os.path.exists(folder_path):
		print(f"Folder {folder_path} not exists")
		return

	for filename in os.listdir(folder_path):
		file_path = os.path.join(folder_path, filename)

		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)

			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print(e)


def load_model_info(model_info_path):
	with open(model_info_path, "r", encoding="utf-8") as f:
		model_info = json.load(f)
	return model_info


def move_config():
	model_info_path = "./data/models_info.json"

	clear_folder("./configs/vocal_models")
	clear_folder("./configs/single_stem_models")
	clear_folder("./configs/multi_stem_models")

	model_info = load_model_info(model_info_path)
	for model_name, model_data in model_info.items():
		config_path = model_data.get("config_path")
		if not config_path:
			print(f"Model {model_name} has no config path")
			continue
		config_path = config_path.replace("configs", "configs_backup")
		with open(config_path, "r", encoding="utf-8") as f:
			config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
			new_config_path = f"./configs/{model_data.get('model_class')}/{model_name}.yaml"
			print(f"Move {config_path} to {new_config_path}")
			save_configs(config, new_config_path)


def save_configs(config, config_path):
	directory = os.path.dirname(config_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	with open(config_path, "w", encoding="utf-8") as f:
		yaml.dump(config.to_dict(), f)


def main():
	move_config()


if __name__ == "__main__":
	main()
