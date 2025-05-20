__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

from utils.constant import *
from webui.utils import i18n, load_configs, get_msst_model, get_vr_model


def init_selected_model():
	try:
		batch_size, num_overlap, chunk_size, is_normalize = None, None, None, False
		config = load_configs(WEBUI_CONFIG)
		selected_model = config["inference"]["selected_model"]
		_, config_path, _, _ = get_msst_model(selected_model)
		config = load_configs(config_path)

		if config.inference.get("batch_size"):
			batch_size = int(config.inference.get("batch_size"))
		if config.inference.get("num_overlap"):
			num_overlap = int(config.inference.get("num_overlap"))
		if config.audio.get("chunk_size"):
			chunk_size = int(config.audio.get("chunk_size"))
		if config.inference.get("normalize"):
			is_normalize = True
		return batch_size, num_overlap, chunk_size, is_normalize
	except:
		return None, None, None, False


def init_selected_msst_model():
	webui_config = load_configs(WEBUI_CONFIG)
	selected_model = webui_config["inference"]["selected_model"]
	insts = [i18n("请先选择模型")]

	if not selected_model:
		return insts

	try:
		_, config_path, _, _ = get_msst_model(selected_model)
		config = load_configs(config_path)
		insts = config.training.instruments
		return insts
	except:
		return insts


def init_selected_vr_model():
	webui_config = load_configs(WEBUI_CONFIG)
	model = webui_config["inference"]["vr_select_model"]
	vr_primary_stem_only = i18n("仅输出主音轨")
	vr_secondary_stem_only = i18n("仅输出次音轨")

	if not model:
		return vr_primary_stem_only, vr_secondary_stem_only

	try:
		primary_stem, secondary_stem, _, _ = get_vr_model(model)
		vr_primary_stem_only = f"{primary_stem} Only"
		vr_secondary_stem_only = f"{secondary_stem} Only"
		return vr_primary_stem_only, vr_secondary_stem_only
	except:
		return vr_primary_stem_only, vr_secondary_stem_only
