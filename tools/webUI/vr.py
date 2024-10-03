import os
import shutil
import threading
import time

from tools.webUI.constant import *
from tools.webUI.utils import i18n, get_vr_model, run_command, load_configs, save_configs

def save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
    config = load_configs(WEBUI_CONFIG)
    config['inference']['vr_select_model'] = vr_select_model
    config['inference']['vr_window_size'] = vr_window_size
    config['inference']['vr_aggression'] = vr_aggression
    config['inference']['vr_output_format'] = vr_output_format
    config['inference']['vr_use_cpu'] = vr_use_cpu
    config['inference']['vr_primary_stem_only'] = vr_primary_stem_only
    config['inference']['vr_secondary_stem_only'] = vr_secondary_stem_only
    config['inference']['vr_multiple_audio_input'] = vr_multiple_audio_input
    config['inference']['vr_store_dir'] = vr_store_dir
    config['inference']['vr_batch_size'] = vr_batch_size
    config['inference']['vr_normalization'] = vr_normalization
    config['inference']['vr_post_process_threshold'] = vr_post_process_threshold
    config['inference']['vr_invert_spect'] = vr_invert_spect
    config['inference']['vr_enable_tta'] = vr_enable_tta
    config['inference']['vr_high_end_process'] = vr_high_end_process
    config['inference']['vr_enable_post_process'] = vr_enable_post_process
    config['inference']['vr_debug_mode'] = vr_debug_mode
    save_configs(config, WEBUI_CONFIG)

def vr_inference_single(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_single_audio, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
    vr_multiple_audio_input = None

    if not vr_single_audio:
        return i18n("请上传至少一个音频文件!")
    if not vr_select_model:
        return i18n("请选择模型")
    if not vr_store_dir:
        return i18n("请选择输出目录")
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    os.makedirs(TEMP_PATH)

    for audio in vr_single_audio:
        shutil.copy(audio, TEMP_PATH)
    
    vr_single_audio = TEMP_PATH
    start_time = time.time()

    save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)

    vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_single_audio, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)
    shutil.rmtree(TEMP_PATH)

    return i18n("运行完成, 耗时: ") + str(round(time.time() - start_time, 2)) + "s"

def vr_inference_multi(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode):
    if not os.path.isdir(vr_multiple_audio_input):
        return i18n("请选择输入文件夹")
    if not vr_select_model:
        return i18n("请选择模型")
    if not vr_store_dir:
        return i18n("请选择输出目录")
    start_time = time.time()

    save_vr_inference_config(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)

    vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode)

    return i18n("运行完成, 耗时: ") + str(round(time.time() - start_time, 2)) + "s"

def vr_inference(vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode, save_another_stem=False, extra_output_dir=None):
    os.makedirs(vr_store_dir, exist_ok=True)

    if extra_output_dir:
        os.makedirs(extra_output_dir, exist_ok=True)

    primary_stem, secondary_stem, _, model_file_dir = get_vr_model(vr_select_model)
    audio_file = vr_audio_input
    model_filename = vr_select_model
    output_format = vr_output_format
    output_dir = vr_store_dir
    invert_spect = "--invert_spect" if vr_invert_spect else ""
    normalization = vr_normalization
    debug_mode = "--debug" if vr_debug_mode else ""

    if vr_primary_stem_only:
        if vr_secondary_stem_only:
            single_stem = ""
        else:
            single_stem = f"--single_stem \"{primary_stem}\""
    else:
        if vr_secondary_stem_only:
            single_stem = f"--single_stem \"{secondary_stem}\""
        else:
            single_stem = ""

    vr_batch_size = int(vr_batch_size)
    vr_aggression = int(vr_aggression)
    use_cpu = "--use_cpu" if vr_use_cpu else ""
    vr_enable_tta = "--vr_enable_tta" if vr_enable_tta else ""
    vr_high_end_process = "--vr_high_end_process" if vr_high_end_process else ""
    vr_enable_post_process = "--vr_enable_post_process" if vr_enable_post_process else ""
    save_another_stem = "--save_another_stem" if save_another_stem else ""
    extra_output_dir = f"--extra_output_dir \"{extra_output_dir}\"" if extra_output_dir else ""

    command = f"{PYTHON} inference/vr_cli.py \"{audio_file}\" {debug_mode} --model_filename \"{model_filename}\" --output_format {output_format} --output_dir \"{output_dir}\" --model_file_dir \"{model_file_dir}\" {invert_spect} --normalization {normalization} {single_stem} {use_cpu} --vr_batch_size {vr_batch_size} --vr_window_size {vr_window_size} --vr_aggression {vr_aggression} {vr_enable_tta} {vr_high_end_process} {vr_enable_post_process} --vr_post_process_threshold {vr_post_process_threshold} {save_another_stem} {extra_output_dir}"

    vr_inference = threading.Thread(target=run_command, args=(command,), name="vr_inference")
    vr_inference.start()
    vr_inference.join()