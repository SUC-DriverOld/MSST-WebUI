import shutil
import requests
import webbrowser
import time
from tqdm import tqdm
import gradio as gr
import traceback

from utils.constant import *
from webui.utils import i18n, load_configs, save_configs, load_vr_model, get_vr_model, load_msst_model, get_msst_model, open_folder, logger

def open_model_folder(model_type):
    if not model_type:
        gr.Info(i18n("请先选择模型类型"))
        return
    if model_type == "UVR_VR_Models":
        config = load_configs(WEBUI_CONFIG)
        uvr_model_folder = config['settings']['uvr_model_dir']
        open_folder(uvr_model_folder)
    else:
        open_folder(os.path.join(MODEL_FOLDER, model_type))

def open_download_manager():
    command = f"{PYTHON} ComfyUI/DownloadManager/main.py"
    logger.info(f"Opening download manager: {command}")
    gr.Info(i18n("已打开下载管理器"))
    os.system(command)

def upgrade_download_model_name(model_type_dropdown):
    if model_type_dropdown == "UVR_VR_Models":
        model_map = load_configs(VR_MODEL)
        return gr.Dropdown(label=i18n("选择模型"), choices=[keys for keys in model_map.keys()])
    else:
        model_map = load_configs(MSST_MODEL)
        return gr.Dropdown(label=i18n("选择模型"), choices=[model["name"] for model in model_map[model_type_dropdown]])

def download_model(model_type, model_name):
    if model_type not in MODEL_CHOICES:
        return i18n("请选择模型类型")

    if model_type == "UVR_VR_Models":
        downloaded_model = load_vr_model()
        if model_name in downloaded_model:
            return i18n("模型") + model_name + i18n("已安装")
        _, _, model_url, model_path = get_vr_model(model_name)
        os.makedirs(model_path, exist_ok=True)
        return download_file(model_url, os.path.join(model_path, model_name), model_name)
    else:
        msst_model_map = load_configs(MSST_MODEL)
        model_mapping = load_msst_model()
        if model_name in model_mapping:
            return i18n("模型") + model_name + i18n("已安装")
        if model_type not in msst_model_map:
            return i18n("模型类型") + model_type + i18n("不存在")
        _, _, _, model_url = get_msst_model(model_name)
        model_path = f"pretrain/{model_type}/{model_name}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        return download_file(model_url, model_path, model_name)

def download_file(url, path, model_name):
    try:
        logger.info(f"Downloading model {model_name} from {url}")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
    
            with open(path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True
            ) as progress_bar:
                last_update_time = time.time()
                bytes_written = 0
                for data in r.iter_content(1024):
                    f.write(data)
                    bytes_written += len(data)
                    current_time = time.time()
                    if current_time - last_update_time >= 1.0:
                        progress_bar.update(bytes_written)
                        bytes_written = 0
                        last_update_time = current_time
                progress_bar.update(bytes_written)

        logger.info(f"Model {model_name} downloaded successfully")

        return i18n("模型") + model_name + i18n("下载成功")
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}\n{traceback.format_exc()}")
        return i18n("模型") + model_name + i18n("下载失败") + str(e)

def manual_download_model(model_type, model_name):
    if model_type not in MODEL_CHOICES:
        return i18n("请选择模型类型")

    if model_type == "UVR_VR_Models":
        downloaded_model = load_vr_model()
        if model_name in downloaded_model:
            return i18n("模型") + model_name + i18n("已安装")
        _, _, model_url, _ = get_vr_model(model_name)
    else:
        vr_model_map = load_configs(MSST_MODEL)
        model_mapping = load_msst_model()
        if model_name in model_mapping:
            return i18n("模型") + model_name + i18n("已安装")
        if model_type not in vr_model_map:
            return i18n("模型类型") + model_type + i18n("不存在")
        _, _, _, model_url = get_msst_model(model_name)

    webbrowser.open(model_url)
    logger.info(f"Opened download link for model {model_name}, link: {model_url}")
    return i18n("已打开") + model_name + i18n("的下载链接")

def update_vr_param(is_BV_model, is_VR51_model, model_param):
    balance_value = gr.Number(label="balance_value", value=0.0, minimum=0.0, maximum=0.9, step=0.1, interactive=True, visible=True if is_BV_model else False)
    out_channels = gr.Number(label="Out Channels", value=32, minimum=1, step=1, interactive=True, visible=True if is_VR51_model else False)
    out_channels_lstm = gr.Number(label="Out Channels (LSTM layer)", value=128, minimum=1, step=1, interactive=True, visible=True if is_VR51_model else False)
    upload_param = gr.File(label=i18n("上传参数文件"), type="filepath", interactive=True, visible=True if model_param == i18n("上传参数") else False)
    return balance_value, out_channels, out_channels_lstm, upload_param

def install_unmsst_model(unmsst_model, unmsst_config, unmodel_class, unmodel_type, unmsst_model_link):
    os.makedirs(os.path.join(UNOFFICIAL_MODEL, "msst_config"), exist_ok=True)

    try:
        model_map = load_configs(os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json"))
    except FileNotFoundError:
        model_map = {"multi_stem_models": [], "single_stem_models": [], "vocal_models": []}

    try:
        model_name = os.path.basename(unmsst_model)

        if model_name in load_msst_model():
            return i18n("模型") + model_name + i18n("已安装")
        if unmsst_model.endswith((".ckpt", ".chpt", ".th")):
            shutil.copy(unmsst_model, os.path.join(MODEL_FOLDER, unmodel_class))
        else: 
            return i18n("请上传'ckpt', 'chpt', 'th'格式的模型文件")
        if unmsst_config.endswith(".yaml"):
            shutil.copy(unmsst_config, os.path.join(UNOFFICIAL_MODEL, "msst_config"))
        else: 
            return i18n("请上传'.yaml'格式的配置文件")

        config = {
            "name": model_name,
            "config_path": os.path.join(UNOFFICIAL_MODEL, "msst_config", os.path.basename(unmsst_config)),
            "model_type": unmodel_type,
            "link": unmsst_model_link
        }

        model_map[unmodel_class].append(config)
        save_configs(model_map, os.path.join(UNOFFICIAL_MODEL, "unofficial_msst_model.json"))
        logger.info(f"Unofficial MSST model {model_name} installed successfully. Model config: {config}")
        return i18n("模型") + os.path.basename(unmsst_model) + i18n("安装成功。重启WebUI以刷新模型列表")
    except Exception as e:
        logger.error(f"Failed to install unofficial MSST model: {str(e)}\n{traceback.format_exc()}")
        return i18n("模型") + os.path.basename(unmsst_model) + i18n("安装失败") + str(e)

def install_unvr_model(unvr_model, unvr_primary_stem, unvr_secondary_stem, model_param, is_karaoke_model, is_BV_model, is_VR51_model, balance_value, out_channels, out_channels_lstm, upload_param, unvr_model_link):
    os.makedirs(UNOFFICIAL_MODEL, exist_ok=True)

    try:
        model_map = load_configs(os.path.join(UNOFFICIAL_MODEL, "unofficial_vr_model.json"))
    except FileNotFoundError:
        model_map = {}

    try:
        model_name = os.path.basename(unvr_model)
        if model_name in load_vr_model():
            return i18n("模型") + model_name + i18n("已安装")
        if unvr_model.endswith(".pth"):
            shutil.copy(unvr_model, "pretrain/VR_Models")
        else: 
            return i18n("请上传'.pth'格式的模型文件")

        if unvr_primary_stem != "" and unvr_secondary_stem != "" and unvr_primary_stem != unvr_secondary_stem:
            model_map[model_name] = {}
            model_map[model_name]["model_path"] = os.path.join(MODEL_FOLDER, "VR_Models", model_name)
            model_map[model_name]["primary_stem"] = unvr_primary_stem
            model_map[model_name]["secondary_stem"] = unvr_secondary_stem
            model_map[model_name]["download_link"] = unvr_model_link
        else: 
            return i18n("请输入正确的音轨名称")

        if model_param == i18n("上传参数"):
            if upload_param.endswith(".json"):
                os.makedirs(os.path.join(UNOFFICIAL_MODEL, "vr_modelparams"), exist_ok=True)
                shutil.copy(upload_param, os.path.join(UNOFFICIAL_MODEL, "vr_modelparams"))
                model_map[model_name]["vr_model_param"] = os.path.basename(upload_param)[:-5]
            else: 
                return i18n("请上传'.json'格式的参数文件")
        else: 
            model_map[model_name]["vr_model_param"] = model_param

        if is_karaoke_model:
            model_map[model_name]["is_karaoke"] = True
        if is_BV_model:
            model_map[model_name]["is_bv_model"] = True
            model_map[model_name]["is_bv_model_rebalanced"] = balance_value
        if is_VR51_model:
            model_map[model_name]["nout"] = out_channels
            model_map[model_name]["nout_lstm"] = out_channels_lstm

        save_configs(model_map, os.path.join(UNOFFICIAL_MODEL, "unofficial_vr_model.json"))
        logger.info(f"Unofficial VR model {model_name} installed successfully. Model config: {model_map[model_name]}")
        return i18n("模型") + os.path.basename(unvr_model) + i18n("安装成功。重启WebUI以刷新模型列表")
    except Exception as e:
        logger.error(f"Failed to install unofficial VR model: {str(e)}\n{traceback.format_exc()}")
        return i18n("模型") + os.path.basename(unvr_model) + i18n("安装失败") + str(e)

def get_all_model_param():
    model_param = [i18n("上传参数")]

    for file in os.listdir(VR_MODELPARAMS):
        if file.endswith(".json"):
            model_param.append(file[:-5])
    return model_param