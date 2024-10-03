import gradio as gr
import pandas as pd

from tools.webUI.constant import *
from tools.webUI.utils import i18n, select_folder, open_folder, stop_all_thread
from tools.webUI.preset import *

def preset(webui_config, presets):
    gr.Markdown(value=i18n("预设流程允许按照预设的顺序运行多个模型。每一个模型的输出将作为下一个模型的输入。"))
    with gr.Tabs():
        with gr.TabItem(label=i18n("使用预设")):
            gr.Markdown(value=i18n("该模式下的UVR推理参数将直接沿用UVR分离页面的推理参数, 如需修改请前往UVR分离页面。<br>修改完成后, 还需要任意处理一首歌才能保存参数! "))
            with gr.Row():
                preset_dropdown = gr.Dropdown(label=i18n("请选择预设"),choices=list(presets.keys()),value=webui_config['inference']['preset'] if webui_config['inference']['preset'] else None, interactive=True, scale=4)
                output_format_flow = gr.Radio(label=i18n("输出格式"),choices=["wav", "mp3", "flac"],value=webui_config['inference']['output_format_flow'] if webui_config['inference']['output_format_flow'] else "wav", interactive=True, scale=1)
            force_cpu = gr.Checkbox(label=i18n("使用CPU (注意: 使用CPU会导致速度非常慢) "),value=webui_config['inference']['force_cpu'] if webui_config['inference']['force_cpu'] else False,interactive=True)
            with gr.Tabs():
                with gr.TabItem(label=i18n("输入音频")):
                    single_audio_flow = gr.Files(label=i18n("上传一个或多个音频文件"), type="filepath")
                with gr.TabItem(label=i18n("输入文件夹")):
                    with gr.Row():
                        input_folder_flow = gr.Textbox(label=i18n("输入目录"),value=webui_config['inference']['input_folder_flow'] if webui_config['inference']['input_folder_flow'] else "input/",interactive=True,scale=3)
                        select_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
                        open_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
            with gr.Row():
                store_dir_flow = gr.Textbox(label=i18n("输出目录"),value=webui_config['inference']['store_dir_flow'] if webui_config['inference']['store_dir_flow'] else "results/",interactive=True,scale=3)
                select_output_dir = gr.Button(i18n("选择文件夹"), scale=1)
                open_output_dir = gr.Button(i18n("打开文件夹"), scale=1)
            with gr.Row():
                single_inference_flow = gr.Button(i18n("输入音频分离"), variant="primary")
                inference_flow = gr.Button(i18n("输入文件夹分离"), variant="primary")
            with gr.Row():
                output_message_flow = gr.Textbox(label="Output Message", scale=4)
                stop_thread = gr.Button(i18n("强制停止"), scale=1)
        with gr.TabItem(label=i18n("制作预设")):
            gr.Markdown(i18n("注意: MSST模型仅支持输出主要音轨, UVR模型支持自定义主要音轨输出。<br>同时输出次级音轨: 选择True将同时输出该次分离得到的次级音轨, **此音轨将直接保存至**输出目录下的secondary_output文件夹, **不会经过后续流程处理**<br>"))
            preset_name_input = gr.Textbox(label=i18n("预设名称"), placeholder=i18n("请输入预设名称"), interactive=True)
            with gr.Row():
                model_type = gr.Dropdown(label=i18n("选择模型类型"), choices=MODEL_CHOICES, interactive=True)
                model_name = gr.Dropdown(label=i18n("选择模型"), choices=[i18n("请先选择模型类型")], interactive=True, scale=2)
                stem = gr.Dropdown(label=i18n("输出音轨"), choices=[i18n("请先选择模型")], interactive=True)
                secondary_output = gr.Dropdown(label=i18n("同时输出次级音轨"), choices=["True", "False"], value="False", interactive=True)
            add_to_flow = gr.Button(i18n("添加至流程"))
            gr.Markdown(i18n("预设流程"))
            preset_flow = gr.Dataframe(pd.DataFrame({"model_type": [""], "model_name": [""], "stem": [""], "secondary_output": [""]}), interactive=False, label=None)
            reset_flow = gr.Button(i18n("重新输入"))
            save_flow = gr.Button(i18n("保存上述预设流程"), variant="primary")
            output_message_make = gr.Textbox(label="Output Message")
        with gr.TabItem(label=i18n("管理预设")):
            gr.Markdown(i18n("此页面提供查看预设, 删除预设, 备份预设, 恢复预设等功能"))
            preset_name_delete = gr.Dropdown(label=i18n("请选择预设"), choices=list(presets.keys()), interactive=True)
            gr.Markdown(i18n("`model_type`: 模型类型；`model_name`: 模型名称；`stem`: 主要输出音轨；<br>`secondary_output`: 同时输出次级音轨。选择True将同时输出该次分离得到的次级音轨, **此音轨将直接保存至**输出目录下的secondary_output文件夹, **不会经过后续流程处理**"))
            preset_flow_delete = gr.Dataframe(pd.DataFrame({"model_type": [i18n("请先选择预设")], "model_name": [i18n("请先选择预设")], "stem": [i18n("请先选择预设")], "secondary_output": [i18n("请先选择预设")]}), interactive=False, label=None)
            delete_button = gr.Button(i18n("删除所选预设"), scale=1)
            gr.Markdown(i18n("每次删除预设前, 将自动备份预设以免误操作。<br>你也可以点击“备份预设流程”按钮进行手动备份, 也可以从备份文件夹中恢复预设流程。"))
            with gr.Row():
                backup_preset = gr.Button(i18n("备份预设流程"))
                open_preset_backup = gr.Button(i18n("打开备份文件夹"))
            with gr.Row():
                select_preset_backup = gr.Dropdown(label=i18n("选择需要恢复的预设流程备份"),choices=preset_backup_list(),interactive=True,scale=4)
                restore_preset = gr.Button(i18n("恢复"), scale=1)
            output_message_manage = gr.Textbox(label="Output Message")
        

    inference_flow.click(fn=run_inference_flow,inputs=[input_folder_flow, store_dir_flow, preset_dropdown, force_cpu, output_format_flow],outputs=output_message_flow)
    single_inference_flow.click(fn=run_single_inference_flow,inputs=[single_audio_flow, store_dir_flow, preset_dropdown, force_cpu, output_format_flow],outputs=output_message_flow)
    select_input_dir.click(fn=select_folder, outputs=input_folder_flow)
    open_input_dir.click(fn=open_folder, inputs=input_folder_flow)
    select_output_dir.click(fn=select_folder, outputs=store_dir_flow)
    open_output_dir.click(fn=open_folder, inputs=store_dir_flow)
    model_type.change(update_model_name, inputs=model_type, outputs=model_name)
    model_name.change(update_model_stem, inputs=[model_type, model_name], outputs=stem)
    add_to_flow.click(add_to_flow_func, [model_type, model_name, stem, secondary_output, preset_flow], preset_flow)
    save_flow.click(save_flow_func, [preset_name_input, preset_flow], [output_message_make, preset_name_delete, preset_dropdown])
    reset_flow.click(reset_flow_func, [], preset_flow)
    delete_button.click(delete_func, [preset_name_delete], [output_message_manage, preset_name_delete, preset_dropdown, preset_flow_delete, select_preset_backup])
    preset_name_delete.change(load_preset, inputs=preset_name_delete, outputs=preset_flow_delete)
    stop_thread.click(fn=stop_all_thread)
    restore_preset.click(fn=restore_preset_func,inputs=[select_preset_backup],outputs=[output_message_manage, preset_dropdown, preset_name_delete, preset_flow_delete])
    backup_preset.click(fn=backup_preset_func,outputs=[output_message_manage, select_preset_backup])
    open_preset_backup.click(open_folder, inputs=gr.Textbox(BACKUP, visible=False))