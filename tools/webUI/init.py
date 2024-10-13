from utils.constant import *
from tools.webUI.utils import i18n, load_configs, get_msst_model, get_vr_model


def init_selected_model():
    try:
        batch_size, dim_t, num_overlap = i18n("该模型不支持修改此值"), i18n("该模型不支持修改此值"), i18n("该模型不支持修改此值")
        config = load_configs(WEBUI_CONFIG)
        selected_model = config['inference']['selected_model']
        _, config_path, _, _ = get_msst_model(selected_model)
        config = load_configs(config_path)

        if config.inference.get('batch_size'):
            batch_size = config.inference.get('batch_size')
        if config.inference.get('dim_t'):
            dim_t = config.inference.get('dim_t')
        if config.inference.get('num_overlap'):
            num_overlap = config.inference.get('num_overlap')
        return batch_size, dim_t, num_overlap
    except: 
        return i18n("请先选择模型"), i18n("请先选择模型"), i18n("请先选择模型")

def init_selected_msst_model():
    webui_config = load_configs(WEBUI_CONFIG)
    selected_model = webui_config['inference']['selected_model']
    extract_instrumental_label = i18n("同时输出次级音轨")
    instrumental_only_label = i18n("仅输出次级音轨")

    if not selected_model:
        return extract_instrumental_label, instrumental_only_label

    try:
        _, config_path, _, _ = get_msst_model(selected_model)
        config = load_configs(config_path)
        target_inst = config.training.get('target_instrument', None)

        if target_inst is None:
            extract_instrumental_label = i18n("此模型默认输出所有音轨")
            instrumental_only_label = i18n("此模型默认输出所有音轨")

        return extract_instrumental_label, instrumental_only_label
    except: 
        return extract_instrumental_label, instrumental_only_label

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