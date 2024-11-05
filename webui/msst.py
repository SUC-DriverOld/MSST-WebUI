import shutil
import time
import gradio as gr
import multiprocessing
import traceback

from utils.constant import *
from webui.utils import i18n, load_configs, save_configs, load_selected_model, logger
from webui.init import get_msst_model
from inference.msst_infer import MSSeparator

def change_to_audio_infer():
    return (gr.Button(i18n("输入音频分离"), variant="primary", visible=True),
            gr.Button(i18n("输入文件夹分离"), variant="primary", visible=False))

def change_to_folder_infer():
    return (gr.Button(i18n("输入音频分离"), variant="primary", visible=False),
            gr.Button(i18n("输入文件夹分离"), variant="primary", visible=True))

def save_model_config(selected_model, batch_size, dim_t, num_overlap, normalize):
    _, config_path, _, _ = get_msst_model(selected_model)
    config = load_configs(config_path)

    if config.inference.get('batch_size'):
        config.inference['batch_size'] = int(batch_size)
    if config.inference.get('dim_t'):
        config.inference['dim_t'] = int(dim_t)
    if config.inference.get('num_overlap'):
        config.inference['num_overlap'] = int(num_overlap)
    if config.inference.get('normalize'):
        config.inference['normalize'] = normalize
    save_configs(config, config_path)
    logger.debug(f"Saved model config: {config['inference']}")
    return i18n("配置保存成功!")

def reset_model_config(selected_model):
    _, original_config_path, _, _ = get_msst_model(selected_model)

    if original_config_path.startswith(UNOFFICIAL_MODEL):
        return i18n("非官方模型不支持重置配置!")

    dir_path, file_name = os.path.split(original_config_path)
    backup_dir_path = dir_path.replace("configs", "configs_backup", 1)
    backup_config_path = os.path.join(backup_dir_path, file_name)

    if os.path.exists(backup_config_path):
        shutil.copy(backup_config_path, original_config_path)
        update_inference_settings(selected_model)
        logger.debug(f"Reset model config: {backup_config_path} -> {original_config_path}")
        return i18n("配置重置成功!")
    else:
        return i18n("备份文件不存在!")

def update_inference_settings(selected_model):
    batch_size = gr.Number(label=i18n("batch_size: 批次大小, 一般不需要改"), interactive=False)
    dim_t = gr.Number(label=i18n("dim_t: 时序维度大小, 一般不需要改"), interactive=False)
    num_overlap = gr.Number(label=i18n("num_overlap: 数值越小速度越快, 但会牺牲效果"), interactive=False)
    normalize = gr.Checkbox(label=i18n("normalize (该模型不支持修改此值) "), value=False, interactive=False)
    extract_instrumental = gr.CheckboxGroup(label=i18n("选择输出音轨"), interactive=False)

    if selected_model:
        _, config_path, _, _ = get_msst_model(selected_model)
        config = load_configs(config_path)

        if config.inference.get('batch_size'):
            batch_size = gr.Number(label=i18n("batch_size: 批次大小, 一般不需要改"), value=int(config.inference.get('batch_size')), interactive=True)
        if config.inference.get('dim_t'):
            dim_t = gr.Number(label=i18n("dim_t: 时序维度大小, 一般不需要改"), value=int(config.inference.get('dim_t')), interactive=True)
        if config.inference.get('num_overlap'):
            num_overlap = gr.Number(label=i18n("num_overlap: 数值越小速度越快, 但会牺牲效果"), value=int(config.inference.get('num_overlap')), interactive=True)
        if config.inference.get('normalize'):
            normalize = gr.Checkbox(label=i18n("normalize: 是否对音频进行归一化处理"), value=config.inference.get('normalize'), interactive=True)
        extract_instrumental = gr.CheckboxGroup(label=i18n("选择输出音轨"), choices=config.training.get('instruments'), interactive=True)

    return batch_size, dim_t, num_overlap, normalize, extract_instrumental

def update_selected_model(model_type):
    webui_config = load_configs(WEBUI_CONFIG)
    webui_config["inference"]["model_type"] = model_type
    save_configs(webui_config, WEBUI_CONFIG)
    return gr.Dropdown(label=i18n("选择模型"), choices=load_selected_model(), value=None, interactive=True, scale=4)

def save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
    config = load_configs(WEBUI_CONFIG)
    config['inference']['selected_model'] = selected_model
    config['inference']['device'] = gpu_id
    config['inference']['output_format'] = output_format
    config['inference']['force_cpu'] = force_cpu
    config['inference']['instrumental'] = extract_instrumental
    config['inference']['use_tta'] = use_tta
    config['inference']['store_dir'] = store_dir
    config['inference']['input_dir'] = input_folder
    save_configs(config, WEBUI_CONFIG)
    logger.debug(f"Saved MSST inference config: {config['inference']}")

def run_inference_single(selected_model, input_audio, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
    input_folder = None

    if not input_audio:
        return i18n("请上传至少一个音频文件!")
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    os.makedirs(TEMP_PATH)

    for audio in input_audio:
        shutil.copy(audio, TEMP_PATH)
    input_path = TEMP_PATH

    save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
    message = start_inference(selected_model, input_path, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
    shutil.rmtree(TEMP_PATH)
    return message

def run_multi_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
    save_msst_inference_config(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)
    return start_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta)

def start_inference(selected_model, input_folder, store_dir, extract_instrumental, gpu_id, output_format, force_cpu, use_tta):
    if selected_model == "":
        return gr.Error(i18n("请选择模型"))
    if input_folder == "":
        return gr.Error(i18n("请选择输入目录"))
    if store_dir == "":
        return gr.Error(i18n("请选择输出目录"))

    gpu_ids = []
    if not force_cpu:
        if len(gpu_id) == 0:
            raise gr.Error(i18n("请选择GPU"))
        try:
            for gpu in gpu_id:
                gpu_ids.append(int(gpu[:gpu.index(":")]))
        except:
            gpu_ids = [0]
    else:
        gpu_ids = [0]

    gpu_ids = list(set(gpu_ids))
    device = "auto" if not force_cpu else "cpu"
    model_path, config_path, model_type, _ = get_msst_model(selected_model)
    webui_config = load_configs(WEBUI_CONFIG)
    debug = webui_config["settings"].get("debug", False)

    if type(store_dir) == str:
        store_dict = {}
        model_config = load_configs(config_path)
        for inst in extract_instrumental:
            if inst in model_config.training.get("instruments"): # bug of gr.CheckboxGroup, we must check if the instrument is in the model
                store_dict[inst] = store_dir
        if store_dict == {}:
            logger.warning(f"No selected instruments, extract all instruments to {store_dir}")
            store_dict = {k : store_dir for k in model_config.training.get("instruments")}
    else:
        store_dict = store_dir

    start_time = time.time()
    logger.info("Starting MSST inference process...")

    result_queue = multiprocessing.Queue()
    msst_inference = multiprocessing.Process(
        target=run_inference,
        args=(model_type, config_path, model_path, device, gpu_ids, output_format, use_tta, store_dict, debug, input_folder, result_queue),
        name="msst_inference"
    )

    msst_inference.start()
    logger.debug(f"Inference process started, PID: {msst_inference.pid}")
    msst_inference.join()

    if not result_queue.empty():
        result = result_queue.get()
        if result[0] == "success":
            return i18n("处理完成, 结果已保存至: ") + store_dir + i18n(", 耗时: ") + \
                str(round(time.time() - start_time, 2)) + "s"
        elif result[0] == "error":
            return i18n("处理失败: ") + result[1]
    else:
        return i18n("用户强制终止")

def run_inference(model_type, config_path, model_path, device, gpu_ids, output_format, use_tta, store_dict, debug, input_folder, result_queue):
    logger.debug(f"Start MSST inference process with parameters: model_type={model_type}, config_path={config_path}, model_path={model_path}, device={device}, gpu_ids={gpu_ids}, output_format={output_format}, use_tta={use_tta}, store_dict={store_dict}, debug={debug}, input_folder={input_folder}")

    try:
        separator = MSSeparator(
            model_type=model_type,
            config_path=config_path,
            model_path=model_path,
            device=device,
            device_ids=gpu_ids,
            output_format=output_format,
            use_tta=use_tta,
            store_dirs=store_dict,
            logger=logger,
            debug=debug
        )
        success_files = separator.process_folder(input_folder)
        separator.del_cache()

        logger.info(f"Successfully separated files: {success_files}")
        result_queue.put(("success", success_files))
    except Exception as e:
        logger.error(f"Separation failed: {str(e)}\n{traceback.format_exc()}")
        result_queue.put(("error", str(e)))

def stop_msst_inference():
    for process in multiprocessing.active_children():
        if process.name == "msst_inference":
            process.terminate()
            process.join()
            logger.info(f"Inference process stopped, PID: {process.pid}")