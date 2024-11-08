import shutil
import time
import gradio as gr
import multiprocessing
import traceback

from utils.constant import *
from webui.utils import i18n, get_vr_model, load_configs, save_configs, logger
from modules.vocal_remover.separator import Separator

def change_to_audio_infer():
    return (gr.Button(i18n("输入音频分离"), variant="primary", visible=True),
            gr.Button(i18n("输入文件夹分离"), variant="primary", visible=False))

def change_to_folder_infer():
    return (gr.Button(i18n("输入音频分离"), variant="primary", visible=False),
            gr.Button(i18n("输入文件夹分离"), variant="primary", visible=True))

def load_vr_model_stem(model):
    primary_stem, secondary_stem, _, _= get_vr_model(model)
    return (gr.Checkbox(label=f"{primary_stem} Only", value=False, interactive=True),
            gr.Checkbox(label=f"{secondary_stem} Only", value=False, interactive=True))

def save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_input, vr_store_dir, vr_batch_size, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process):
    config = load_configs(WEBUI_CONFIG)
    config['inference']['vr_select_model'] = vr_select_model
    config['inference']['vr_window_size'] = int(vr_window_size)
    config['inference']['vr_aggression'] = int(vr_aggression)
    config['inference']['output_format'] = vr_output_format
    config['inference']['force_cpu'] = vr_use_cpu
    config['inference']['vr_primary_stem_only'] = vr_primary_stem_only
    config['inference']['vr_secondary_stem_only'] = vr_secondary_stem_only
    config['inference']['input_dir'] = vr_input
    config['inference']['store_dir'] = vr_store_dir
    config['inference']['vr_batch_size'] = int(vr_batch_size)
    config['inference']['vr_post_process_threshold'] = float(vr_post_process_threshold)
    config['inference']['vr_invert_spect'] = vr_invert_spect
    config['inference']['vr_enable_tta'] = vr_enable_tta
    config['inference']['vr_high_end_process'] = vr_high_end_process
    config['inference']['vr_enable_post_process'] = vr_enable_post_process
    save_configs(config, WEBUI_CONFIG)
    logger.debug(f"Saved VR inference config: {config['inference']}")

def vr_inference_single(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, audio_input, vr_store_dir, vr_batch_size, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process):
    vr_input_save = None

    if not audio_input:
        return i18n("请上传至少一个音频文件!")
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    os.makedirs(TEMP_PATH)

    for audio in audio_input:
        shutil.copy(audio, TEMP_PATH)
    vr_audio_input = TEMP_PATH

    save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_input_save, vr_store_dir, vr_batch_size, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process)
    message = start_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_audio_input, vr_store_dir, vr_batch_size, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process)
    shutil.rmtree(TEMP_PATH)
    return message

def vr_inference_multi(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, folder_input, vr_store_dir, vr_batch_size, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process):
    save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, folder_input, vr_store_dir, vr_batch_size, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process)
    return start_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, folder_input, vr_store_dir, vr_batch_size, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process)

def start_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, audio_input, vr_store_dir, vr_batch_size, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process):
    if not audio_input:
        return i18n("请上传至少一个音频文件!")
    if not vr_select_model:
        return i18n("请选择模型")
    if not vr_store_dir:
        return i18n("请选择输出目录")

    webui_config = load_configs(WEBUI_CONFIG)
    debug = webui_config["settings"].get("debug", False)
    primary_stem, secondary_stem, _, model_path = get_vr_model(vr_select_model)
    model_file = os.path.join(model_path, vr_select_model)

    if not vr_primary_stem_only and not vr_secondary_stem_only:
        logger.warning("Both primary and secondary stem are disabled, enable both by default")
        vr_primary_stem_only = True
        vr_secondary_stem_only = True

    output_dir = {
        primary_stem: vr_store_dir if vr_primary_stem_only else "",
        secondary_stem: vr_store_dir if vr_secondary_stem_only else ""
    }

    start_time = time.time()
    logger.info("Straring VR inference process...")

    result_queue = multiprocessing.Queue()
    vr_inference = multiprocessing.Process(
        target=run_inference,
        args=(debug, model_file, output_dir, vr_output_format, vr_invert_spect, vr_use_cpu, int(vr_batch_size), int(vr_window_size), int(vr_aggression), vr_enable_tta, vr_enable_post_process, vr_post_process_threshold, vr_high_end_process, audio_input, result_queue),
        name="vr_inference"
    )

    vr_inference.start()
    logger.debug(f"Inference process started, PID: {vr_inference.pid}")
    vr_inference.join()

    if not result_queue.empty():
        result = result_queue.get()
        if result[0] == "success":
            return i18n("处理完成, 结果已保存至: ") + vr_store_dir + i18n(", 耗时: ") + \
                str(round(time.time() - start_time, 2)) + "s"
        elif result[0] == "error":
            return i18n("处理失败: ") + result[1]
    else:
        return i18n("用户强制终止")

def run_inference(debug, model_file, output_dir, output_format, invert_using_spec, use_cpu, batch_size, window_size, aggression, enable_tta, enable_post_process, post_process_threshold, high_end_process, input_folder, result_queue):
    logger.debug(f"Start VR inference process with parameters: debug={debug}, model_file={model_file}, output_dir={output_dir}, output_format={output_format}, invert_using_spec={invert_using_spec}, use_cpu={use_cpu}, batch_size={batch_size}, window_size={window_size}, aggression={aggression}, enable_tta={enable_tta}, enable_post_process={enable_post_process}, post_process_threshold={post_process_threshold}, high_end_process={high_end_process}, input_folder={input_folder}")

    try:
        separator = Separator(
            logger=logger,
            debug=debug,
            model_file=model_file,
            output_dir=output_dir,
            output_format=output_format,
            invert_using_spec=invert_using_spec,
            use_cpu=use_cpu,
            vr_params={
                "batch_size": batch_size,
                "window_size": window_size,
                "aggression": aggression,
                "enable_tta": enable_tta,
                "enable_post_process": enable_post_process,
                "post_process_threshold": post_process_threshold,
                "high_end_process": high_end_process
            },
        )
        success_files = separator.process_folder(input_folder)
        separator.del_cache()

        logger.info(f"Successfully separated files: {success_files}")
        result_queue.put(("success", success_files))
    except Exception as e:
        logger.error(f"Separation failed: {str(e)}\n{traceback.format_exc()}")
        result_queue.put(("error", str(e)))

def stop_vr_inference():
    for process in multiprocessing.active_children():
        if process.name == "vr_inference":
            process.terminate()
            process.join()
            logger.info(f"Inference process stopped, PID: {process.pid}")