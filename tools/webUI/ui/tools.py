import gradio as gr

from utils.constant import *
from tools.webUI.utils import i18n, select_folder, open_folder
from tools.webUI.tools import *

def tools(webui_config):
    with gr.Tabs():
        with gr.TabItem(label=i18n("音频格式转换")):
            gr.Markdown(value=i18n("上传一个或多个音频文件并将其转换为指定格式。<br>支持的格式包括 .mp3, .flac, .wav, .ogg, .m4a, .wma, .aac...等等。<br>**不支持**网易云音乐/QQ音乐等加密格式, 如.ncm, .qmc等。"))
            with gr.Row():
                inputs = gr.Files(label=i18n("上传一个或多个音频文件"))
                with gr.Column():
                    ffmpeg_output_format = gr.Dropdown(label=i18n("选择或输入音频输出格式"),choices=["wav", "flac", "mp3", "ogg", "m4a", "wma", "aac"],value=webui_config['tools']['ffmpeg_output_format'] if webui_config['tools']['ffmpeg_output_format'] else "wav",allow_custom_value=True,interactive=True)
                    ffmpeg_output_folder = gr.Textbox(label=i18n("选择音频输出目录"), value=webui_config['tools']['ffmpeg_output_folder'] if webui_config['tools']['ffmpeg_output_folder'] else "results/ffmpeg_output/", interactive=True)
                    with gr.Row():
                        select_ffmpeg_output_dir = gr.Button(i18n("选择文件夹"))
                        open_ffmpeg_output_dir = gr.Button(i18n("打开文件夹"))
            convert_audio_button = gr.Button(i18n("转换音频"), variant="primary")
            output_message_ffmpeg = gr.Textbox(label="Output Message")
        with gr.TabItem(label=i18n("合并音频")):
            gr.Markdown(value=i18n("点击合并音频按钮后, 将自动把输入文件夹中的所有音频文件合并为一整个音频文件<br>目前支持的格式包括 .mp3, .flac, .wav, .ogg 这四种<br>合并后的音频会保存至输出目录中, 文件名为merged_audio.wav"))
            with gr.Row():
                merge_audio_input = gr.Textbox(label=i18n("输入目录"),value=webui_config['tools']['merge_audio_input'] if webui_config['tools']['merge_audio_input'] else "input/",interactive=True,scale=3)
                select_merge_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
                open_merge_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
            with gr.Row():
                merge_audio_output = gr.Textbox(label=i18n("输出目录"),value=webui_config['tools']['merge_audio_output'] if webui_config['tools']['merge_audio_output'] else "results/merge",interactive=True,scale=3)
                select_merge_output_dir = gr.Button(i18n("选择文件夹"), scale=1)
                open_merge_output_dir = gr.Button(i18n("打开文件夹"), scale=1)
            merge_audio_button = gr.Button(i18n("合并音频"), variant="primary")
            output_message_merge = gr.Textbox(label="Output Message")
        with gr.TabItem(label=i18n("计算SDR")):
            with gr.Column():
                gr.Markdown(value=i18n("上传两个**wav音频文件**并计算它们的[SDR](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021#evaluation-metric)。<br>SDR是一个用于评估模型质量的数值。数值越大, 模型算法结果越好。"))
            with gr.Row():
                reference_audio = gr.File(label=i18n("原始音频"), type="filepath")
                estimated_audio = gr.File(label=i18n("分离后的音频"), type="filepath")
            compute_sdr_button = gr.Button(i18n("计算SDR"), variant="primary")
            output_message_sdr = gr.Textbox(label="Output Message")
        with gr.TabItem(label = i18n("Ensemble模式")):
            gr.Markdown(value = i18n("可用于集成不同算法的结果。具体的文档位于/docs/ensemble.md"))
            with gr.Row():
                files = gr.Files(label = i18n("上传多个音频文件"), type = "filepath", file_count = 'multiple')
                with gr.Column():
                    with gr.Row():
                        ensemble_type = gr.Dropdown(choices = ["avg_wave", "median_wave", "min_wave", "max_wave", "avg_fft", "median_fft", "min_fft", "max_fft"],label = i18n("集成模式"),value = webui_config['tools']['ensemble_type'] if webui_config['tools']['ensemble_type'] else "avg_wave",interactive=True)
                        weights = gr.Textbox(label = i18n("权重(以空格分隔, 数量要与上传的音频一致)"), value = "1 1")
                    ensembl_output_path = gr.Textbox(label = i18n("输出目录"), value = webui_config['tools']['ensemble_output_folder'] if webui_config['tools']['ensemble_output_folder'] else "results/ensemble/",interactive=True)
                    with gr.Row():
                        select_ensembl_output_path = gr.Button(i18n("选择文件夹"))
                        open_ensembl_output_path = gr.Button(i18n("打开文件夹"))
            ensemble_button = gr.Button(i18n("运行"), variant = "primary")
            output_message_ensemble = gr.Textbox(label = "Output Message")
            with gr.Row():
                with gr.Column():
                    gr.Markdown(i18n("### 集成模式"))
                    gr.Markdown(i18n("1. `avg_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的平均值<br>2. `median_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的中位数<br>3. `min_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的最小绝对值<br>4. `max_wave`: 在1D变体上进行集成, 独立地找到波形的每个样本的最大绝对值<br>5. `avg_fft`: 在频谱图 (短时傅里叶变换 (STFT) 2D变体) 上进行集成, 独立地找到频谱图的每个像素的平均值。平均后使用逆STFT得到原始的1D波形<br>6. `median_fft`: 与avg_fft相同, 但使用中位数代替平均值 (仅在集成3个或更多来源时有用) <br>7. `min_fft`: 与avg_fft相同, 但使用最小函数代替平均值 (减少激进程度) <br>8. `max_fft`: 与avg_fft相同, 但使用最大函数代替平均值 (增加激进程度) "))
                with gr.Column():
                    gr.Markdown(i18n("### 注意事项"))
                    gr.Markdown(i18n("1. min_fft可用于进行更保守的合成, 它将减少更激进模型的影响。<br>2. 最好合成等质量的模型。在这种情况下, 它将带来增益。如果其中一个模型质量不好, 它将降低整体质量。<br>3. 在原仓库作者的实验中, 与其他方法相比, avg_wave在SDR分数上总是更好或相等。<br>4. 最终会在输出目录下生成一个`ensemble_<集成模式>.wav`。"))
        with gr.TabItem(label=i18n("歌声转MIDI")):
            gr.Markdown(value=i18n("歌声转MIDI功能使用开源项目[SOME](https://github.com/openvpi/SOME/), 可以将分离得到的**干净的歌声**转换成.mid文件。<br>【必须】若想要使用此功能, 请先下载权重文件[model_steps_64000_simplified.ckpt](https://hf-mirror.com/Sucial/SOME_Models/resolve/main/model_steps_64000_simplified.ckpt)并将其放置在程序目录下的`tools/SOME_weights`文件夹内。文件命名不可随意更改! <br>【重要】只能上传wav格式的音频! "))
            gr.Markdown(value=i18n("如果不知道如何测量歌曲BPM, 可以尝试这两个在线测量工具: [bpmdetector](https://bpmdetector.kniffen.dev/) | [key-bpm-finder](https://vocalremover.org/zh/key-bpm-finder), 测量时建议上传原曲或伴奏, 若干声可能导致测量结果不准确。"))
            with gr.Row():
                some_input_audio = gr.File(label=i18n("上传wav格式音频"), type="filepath")
                with gr.Column():
                    audio_bpm = gr.Number(label=i18n("输入音频BPM"), value=120, interactive=True)
                    some_output_folder = gr.Textbox(label=i18n("输出目录"),value=webui_config['tools']['some_output_folder'] if webui_config['tools']['some_output_folder'] else "results/some/",interactive=True,scale=3)
                    with gr.Row():
                        select_some_output_dir = gr.Button(i18n("选择文件夹"))
                        open_some_output_dir = gr.Button(i18n("打开文件夹"))
            some_button = gr.Button(i18n("开始转换"), variant="primary")
            output_message_some = gr.Textbox(label="Output Message")
            gr.Markdown(i18n("### 注意事项"))
            gr.Markdown(i18n("1. 音频BPM (每分钟节拍数) 可以通过MixMeister BPM Analyzer等软件测量获取。<br>2. 为保证MIDI提取质量, 音频文件请采用干净清晰无混响底噪人声。<br>3. 输出MIDI不带歌词信息, 需要用户自行添加歌词。<br>4. 实际使用体验中部分音符会出现断开的现象, 需自行修正。SOME的模型主要面向DiffSinger唱法模型自动标注, 比正常用户在创作中需要的MIDI更加精细, 因而可能导致模型倾向于对音符进行切分。<br>5. 提取的MIDI没有量化/没有对齐节拍/不适配BPM, 需自行到各编辑器中手动调整。"))

    convert_audio_button.click(fn=convert_audio, inputs=[inputs, ffmpeg_output_format, ffmpeg_output_folder], outputs=output_message_ffmpeg)
    select_ffmpeg_output_dir.click(fn=select_folder, outputs=ffmpeg_output_folder)
    open_ffmpeg_output_dir.click(fn=open_folder, inputs=ffmpeg_output_folder)
    merge_audio_button.click(merge_audios, [merge_audio_input, merge_audio_output], outputs=output_message_merge)
    select_merge_input_dir.click(fn=select_folder, outputs=merge_audio_input)
    open_merge_input_dir.click(fn=open_folder, inputs=merge_audio_input)
    select_merge_output_dir.click(fn=select_folder, outputs=merge_audio_output)
    open_merge_output_dir.click(fn=open_folder, inputs=merge_audio_output)
    compute_sdr_button.click(process_audio, [reference_audio, estimated_audio], outputs=output_message_sdr)
    ensemble_button.click(fn = ensemble, inputs = [files, ensemble_type, weights, ensembl_output_path],outputs = output_message_ensemble)
    select_ensembl_output_path.click(fn=select_folder, outputs = ensembl_output_path)
    open_ensembl_output_path.click(fn=open_folder, inputs = ensembl_output_path)
    select_some_output_dir.click(fn=select_folder, outputs=some_output_folder)
    open_some_output_dir.click(fn=open_folder, inputs=some_output_folder)
    some_button.click(fn=some_inference,inputs=[some_input_audio, audio_bpm, some_output_folder],outputs=output_message_some)