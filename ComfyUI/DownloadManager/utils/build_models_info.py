import json
from huggingface_hub import HfApi


def get_file_info_from_hub(repo_id, filename):
	api = HfApi()
	try:
		model_info = api.model_info(repo_id, files_metadata=True)
		if not model_info.siblings:
			return None

		for sibling in model_info.siblings:
			if sibling.rfilename == filename:
				if hasattr(sibling, "lfs") and sibling.lfs:
					print(f"Found {filename} in Hugging Face Hub.")
					return {"sha256": sibling.lfs.sha256, "size": sibling.lfs.size}
				else:
					return {"sha256": getattr(sibling, "sha256", None), "size": getattr(sibling, "size", None)}

		return None
	except Exception as e:
		return None


def create_models_info():
	repo_id = "Sucial/MSST-WebUI"
	models_info = {}

	try:
		with open("./data/msst_model_map.json", "r", encoding="utf-8") as f:
			msst_map = json.load(f)
		with open("./data/vr_model_map.json", "r", encoding="utf-8") as f:
			vr_map = json.load(f)
	except Exception as e:
		return

	test_info = get_file_info_from_hub(repo_id, "All_Models/VR_Models/2_HP-UVR.pth")
	if test_info is None:
		return

	for model_name, model_data in vr_map.items():
		filepath = f"All_Models/VR_Models/{model_name}"
		info = get_file_info_from_hub(repo_id, filepath)

		if info:
			is_karaoke_model = model_data.get("is_karaoke", False)
			is_BV_model = model_data.get("is_bv_model", False)
			is_VR51_model = "nout" in model_data and "nout_lstm" in model_data
			models_info[model_name] = {
				"model_class": "VR_Models",
				"model_name": model_name,
				"model_size": info["size"],
				"sha256": info["sha256"],
				"is_installed": False,
				"target_position": f"./pretrain/VR_Models/{model_name}",
				"primary_stem": model_data["primary_stem"],
				"secondary_stem": model_data["secondary_stem"],
				"vr_model_param": model_data["vr_model_param"],
				"link": model_data["download_link"],
			}
			if is_karaoke_model:
				models_info[model_name]["is_karaoke"] = True
			if is_BV_model:
				models_info[model_name]["is_bv_model"] = True
				models_info[model_name]["is_bv_model_rebalanced"] = model_data["is_bv_model_rebalanced"]
			if is_VR51_model:
				models_info[model_name]["nout"] = model_data["nout"]
				models_info[model_name]["nout_lstm"] = model_data["nout_lstm"]
		else:
			pass

	categories = ["multi_stem_models", "single_stem_models", "vocal_models"]

	for category in categories:
		if category in msst_map:
			for model in msst_map[category]:
				model_name = model["name"]
				filepath = f"All_Models/{category}/{model_name}"
				info = get_file_info_from_hub(repo_id, filepath)

				if info:
					models_info[model_name] = {
						"model_class": category,
						"model_name": model_name,
						"model_size": info["size"],
						"sha256": info["sha256"],
						"is_installed": False,
						"target_position": f"./pretrain/{category}/{model_name}",
						# "config_path": model["config_path"],
						"model_type": model["model_type"],
						"link": model["link"],
					}

	try:
		with open("models_info.json", "w", encoding="utf-8") as f:
			json.dump(models_info, f, indent=4, ensure_ascii=False)
	except Exception as e:
		pass


if __name__ == "__main__":
	create_models_info()
