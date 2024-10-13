import gradio as gr

from utils.constant import *
from tools.webUI.utils import i18n, load_vr_model, load_vr_model_stem, stop_all_thread, select_folder, open_folder
from tools.webUI.init import init_selected_vr_model
from tools.webUI.vr import vr_inference_single, vr_inference_multi

def vr(webui_config):
    gr.Markdown(value=i18n("说明: 本整合包仅融合了UVR的VR Architecture模型, MDX23C和HtDemucs类模型可以直接使用前面的MSST音频分离。<br>UVR分离使用项目: [https://github.com/nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 并进行了优化。"))
    vr_select_model = gr.Dropdown(label=i18n("选择模型"), choices=load_vr_model(), value=webui_config['inference']['vr_select_model'] if webui_config['inference']['vr_select_model'] else None,interactive=True)
    with gr.Row():
        vr_window_size = gr.Dropdown(label=i18n("Window Size: 窗口大小, 用于平衡速度和质量"), choices=["320", "512", "1024"], value=webui_config['inference']['vr_window_size'] if webui_config['inference']['vr_window_size'] else "512", interactive=True)
        vr_aggression = gr.Number(label=i18n("Aggression: 主干提取强度, 范围-100-100, 人声请选5"), minimum=-100, maximum=100, value=webui_config['inference']['vr_aggression'] if webui_config['inference']['vr_aggression'] else 5, interactive=True)
        vr_output_format = gr.Radio(label=i18n("输出格式"), choices=["wav", "flac", "mp3"], value=webui_config['inference']['vr_output_format'] if webui_config['inference']['vr_output_format'] else "wav", interactive=True)
    with gr.Row():
        vr_primary_stem_label, vr_secondary_stem_label = init_selected_vr_model()
        vr_use_cpu = gr.Checkbox(label=i18n("使用CPU"), value=webui_config['inference']['vr_use_cpu'] if webui_config['inference']['vr_use_cpu'] else False, interactive=True)
        vr_primary_stem_only = gr.Checkbox(label=vr_primary_stem_label, value=webui_config['inference']['vr_primary_stem_only'] if webui_config['inference']['vr_primary_stem_only'] else False, interactive=True)
        vr_secondary_stem_only = gr.Checkbox(label=vr_secondary_stem_label, value=webui_config['inference']['vr_secondary_stem_only'] if webui_config['inference']['vr_secondary_stem_only'] else False, interactive=True)
    with gr.Tabs():
        with gr.TabItem(label=i18n("输入音频")):
            vr_single_audio = gr.Files(label="上传一个或多个音频文件", type="filepath")
        with gr.TabItem(label=i18n("输入文件夹")):
            with gr.Row():
                vr_multiple_audio_input = gr.Textbox(label=i18n("输入目录"),value=webui_config['inference']['vr_multiple_audio_input'] if webui_config['inference']['vr_multiple_audio_input'] else "input/", interactive=True, scale=3)
                vr_select_multi_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
                vr_open_multi_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
    with gr.Row():
        vr_store_dir = gr.Textbox(label=i18n("输出目录"),value=webui_config['inference']['vr_store_dir'] if webui_config['inference']['vr_store_dir'] else "results/",interactive=True,scale=3)
        vr_select_store_btn = gr.Button(i18n("选择文件夹"), scale=1)
        vr_open_store_btn = gr.Button(i18n("打开文件夹"), scale=1)
    with gr.Accordion(i18n("以下是一些高级设置, 一般保持默认即可"), open=False):
        with gr.Row():
            vr_batch_size = gr.Number(label=i18n("Batch Size: 一次要处理的批次数, 越大占用越多RAM, 处理速度加快"), minimum=1, value=webui_config['inference']['vr_batch_size'] if webui_config['inference']['vr_batch_size'] else 2, interactive=True)
            vr_normalization = gr.Number(label=i18n("Normalization: 最大峰值振幅, 用于归一化输入和输出音频。取值为0-1"), minimum=0.0, maximum=1.0, step=0.01, value=webui_config['inference']['vr_normalization'] if webui_config['inference']['vr_normalization'] else 1, interactive=True)
            vr_post_process_threshold = gr.Number(label=i18n("Post Process Threshold: 后处理特征阈值, 取值为0.1-0.3"), minimum=0.1, maximum=0.3, step=0.01, value=webui_config['inference']['vr_post_process_threshold'] if webui_config['inference']['vr_post_process_threshold'] else 0.2, interactive=True)
        with gr.Row():
            vr_invert_spect = gr.Checkbox(label=i18n("Invert Spectrogram: 二级步骤将使用频谱图而非波形进行反转, 可能会提高质量, 但速度稍慢"), value=webui_config['inference']['vr_invert_spect'] if webui_config['inference']['vr_invert_spect'] else False, interactive=True)
            vr_enable_tta = gr.Checkbox(label=i18n("Enable TTA: 启用“测试时增强”, 可能会提高质量, 但速度稍慢"), value=webui_config['inference']['vr_enable_tta'] if webui_config['inference']['vr_enable_tta'] else False, interactive=True)
            vr_high_end_process = gr.Checkbox(label=i18n("High End Process: 将输出音频缺失的频率范围镜像输出"), value=webui_config['inference']['vr_high_end_process'] if webui_config['inference']['vr_high_end_process'] else False, interactive=True)
            vr_enable_post_process = gr.Checkbox(label=i18n("Enable Post Process: 识别人声输出中残留的人工痕迹, 可改善某些歌曲的分离效果"), value=webui_config['inference']['vr_enable_post_process'] if webui_config['inference']['vr_enable_post_process'] else False, interactive=True)
        vr_debug_mode = gr.Checkbox(label=i18n("Debug Mode: 启用调试模式, 向开发人员反馈时, 请开启此模式"), value=webui_config['inference']['vr_debug_mode'] if webui_config['inference']['vr_debug_mode'] else False, interactive=True)
    with gr.Row():
        vr_start_single_inference = gr.Button(i18n("输入音频分离"), variant="primary")
        vr_start_multi_inference = gr.Button(i18n("输入文件夹分离"), variant="primary")
    with gr.Row():
        vr_output_message = gr.Textbox(label="Output Message", scale=4)
        stop_thread = gr.Button(i18n("强制停止"), scale=1)

    vr_select_model.change(fn=load_vr_model_stem,inputs=vr_select_model,outputs=[vr_primary_stem_only, vr_secondary_stem_only])
    vr_select_multi_input_dir.click(fn=select_folder, outputs=vr_multiple_audio_input)
    vr_open_multi_input_dir.click(fn=open_folder, inputs=vr_multiple_audio_input)
    vr_select_store_btn.click(fn=select_folder, outputs=vr_store_dir)
    vr_open_store_btn.click(fn=open_folder, inputs=vr_store_dir)
    vr_start_single_inference.click(fn=vr_inference_single,inputs=[vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_single_audio, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode],outputs=vr_output_message)
    vr_start_multi_inference.click(fn=vr_inference_multi,inputs=[vr_select_model, vr_window_size, vr_aggression, vr_output_format, vr_use_cpu, vr_primary_stem_only, vr_secondary_stem_only, vr_multiple_audio_input, vr_store_dir, vr_batch_size, vr_normalization, vr_post_process_threshold, vr_invert_spect, vr_enable_tta, vr_high_end_process, vr_enable_post_process, vr_debug_mode],outputs=vr_output_message)
    stop_thread.click(fn=stop_all_thread)
