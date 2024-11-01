from utils.constant import *
from webui.utils import i18n, load_configs, get_msst_model, get_vr_model


def init_selected_model():
    try:
        batch_size, dim_t, num_overlap, is_normalize = (
            i18n("该模型不支持修改此值"),
            i18n("该模型不支持修改此值"),
            i18n("该模型不支持修改此值"),
            False
        )
        config = load_configs(WEBUI_CONFIG)
        selected_model = config['inference']['selected_model']
        _, config_path, _, _ = get_msst_model(selected_model)
        config = load_configs(config_path)

        if config.inference.get('batch_size'):
            batch_size = int(config.inference.get('batch_size'))
        if config.inference.get('dim_t'):
            dim_t = int(config.inference.get('dim_t'))
        if config.inference.get('num_overlap'):
            num_overlap = int(config.inference.get('num_overlap'))
        if config.inference.get('normalize'):
            is_normalize = True
        return batch_size, dim_t, num_overlap, is_normalize
    except: 
        return i18n("请先选择模型"), i18n("请先选择模型"), i18n("请先选择模型"), False

def init_selected_msst_model():
    webui_config = load_configs(WEBUI_CONFIG)
    selected_model = webui_config['inference']['selected_model']
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
    model = webui_config['inference']['vr_select_model']
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