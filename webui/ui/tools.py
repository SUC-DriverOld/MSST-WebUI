__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr

from webui.utils import i18n, select_folder, open_folder
from webui.tools import convert_audio, merge_audios, caculate_sdr, some_inference


def tools(webui_config):
	channels = webui_config["tools"]["channels"]
	if channels == 1:
		channels = i18n("单声道")
	else:
		channels = i18n("立体声")

	with gr.Tabs():
		with gr.TabItem(label=i18n("音频格式转换")):
			gr.Markdown(
				value=i18n(
					"上传一个或多个音频文件并将其转换为指定格式。<br>支持的格式包括 .mp3, .flac, .wav, .ogg, .m4a, .wma, .aac...等等。<br>**不支持**网易云音乐/QQ音乐等加密格式, 如.ncm, .qmc等。"
				)
			)
			with gr.Row():
				inputs = gr.Files(label=i18n("上传一个或多个音频文件"))
				with gr.Column():
					ffmpeg_output_format = gr.Dropdown(
						label=i18n("选择或输入音频输出格式"),
						choices=["wav", "flac", "mp3", "ogg", "m4a", "wma", "aac"],
						value=webui_config["tools"]["output_format"] if webui_config["tools"]["output_format"] else "wav",
						allow_custom_value=True,
						interactive=True,
					)
					ffmpeg_output_folder = gr.Textbox(label=i18n("选择音频输出目录"), value=webui_config["tools"]["store_dir"] if webui_config["tools"]["store_dir"] else "results/", interactive=True)
					with gr.Row():
						select_ffmpeg_output_dir = gr.Button(i18n("选择文件夹"))
						open_ffmpeg_output_dir = gr.Button(i18n("打开文件夹"))
			with gr.Row():
				sample_rate = gr.Radio(
					label=i18n("输出音频采样率(Hz)"), choices=[32000, 44100, 48000], value=webui_config["tools"]["sample_rate"] if webui_config["tools"]["sample_rate"] else 44100, interactive=True
				)
				channels = gr.Radio(label=i18n("输出音频声道数"), choices=[i18n("单声道"), i18n("立体声")], value=channels, interactive=True)
			with gr.Row():
				wav_bit_depth = gr.Radio(
					label=i18n("输出wav位深度"),
					choices=["PCM-16", "PCM-24", "PCM-32"],
					value=webui_config["tools"]["wav_bit_depth"] if webui_config["tools"]["wav_bit_depth"] else "PCM-16",
					interactive=True,
				)
				flac_bit_depth = gr.Radio(
					label=i18n("输出flac位深度"), choices=["16-bit", "32-bit"], value=webui_config["tools"]["flac_bit_depth"] if webui_config["tools"]["flac_bit_depth"] else "16-bit", interactive=True
				)
				mp3_bit_rate = gr.Radio(
					label=i18n("输出mp3比特率(bps)"),
					choices=["192k", "256k", "320k"],
					value=webui_config["tools"]["mp3_bit_rate"] if webui_config["tools"]["mp3_bit_rate"] else "320k",
					interactive=True,
				)
				ogg_bit_rate = gr.Radio(
					label=i18n("输出ogg比特率(bps)"),
					choices=["192k", "320k", "450k"],
					value=webui_config["tools"]["ogg_bit_rate"] if webui_config["tools"]["ogg_bit_rate"] else "320k",
					interactive=True,
				)
			convert_audio_button = gr.Button(i18n("转换音频"), variant="primary")
			output_message_ffmpeg = gr.Textbox(label="Output Message")
		with gr.TabItem(label=i18n("合并音频")):
			gr.Markdown(value=i18n("点击合并音频按钮后, 将自动把输入文件夹中的所有音频文件合并为一整个音频文件<br>合并后的音频会保存至输出目录中, 文件名为merged_audio_<文件夹名字>.wav"))
			with gr.Row():
				merge_audio_input = gr.Textbox(
					label=i18n("输入目录"), value=webui_config["tools"]["merge_audio_input"] if webui_config["tools"]["merge_audio_input"] else "input/", interactive=True, scale=4
				)
				select_merge_input_dir = gr.Button(i18n("选择文件夹"), scale=1)
				open_merge_input_dir = gr.Button(i18n("打开文件夹"), scale=1)
			with gr.Row():
				merge_audio_output = gr.Textbox(label=i18n("输出目录"), value=webui_config["tools"]["store_dir"] if webui_config["tools"]["store_dir"] else "results/", interactive=True, scale=4)
				select_merge_output_dir = gr.Button(i18n("选择文件夹"), scale=1)
				open_merge_output_dir = gr.Button(i18n("打开文件夹"), scale=1)
			merge_audio_button = gr.Button(i18n("合并音频"), variant="primary")
			output_message_merge = gr.Textbox(label="Output Message")
		with gr.TabItem(label=i18n("计算SDR")):
			with gr.Column():
				gr.Markdown(
					value=i18n(
						"上传两个**wav音频文件**并计算它们的[SDR](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021#evaluation-metric)。<br>SDR是一个用于评估模型质量的数值。数值越大, 模型算法结果越好。"
					)
				)
			with gr.Row():
				reference_audio = gr.File(label=i18n("参考音频"), type="filepath")
				estimated_audio = gr.File(label=i18n("待估音频"), type="filepath")
			compute_sdr_button = gr.Button(i18n("计算SDR"), variant="primary")
			output_message_sdr = gr.Textbox(label="Output Message")
		with gr.TabItem(label=i18n("歌声转MIDI")):
			gr.Markdown(
				value=i18n(
					"歌声转MIDI功能使用开源项目[SOME](https://github.com/openvpi/SOME/), 可以将分离得到的**干净的歌声**转换成.mid文件。<br>【必须】若想要使用此功能, 请先下载权重文件[model_steps_64000_simplified.ckpt](https://hf-mirror.com/Sucial/MSST-WebUI/resolve/main/SOME_weights/model_steps_64000_simplified.ckpt)并将其放置在程序目录下的`tools/SOME_weights`文件夹内。文件命名不可随意更改!"
				)
			)
			gr.Markdown(
				value=i18n(
					"如果不知道如何测量歌曲BPM, 可以尝试这两个在线测量工具: [bpmdetector](https://bpmdetector.kniffen.dev/) | [key-bpm-finder](https://vocalremover.org/zh/key-bpm-finder), 测量时建议上传原曲或伴奏, 若干声可能导致测量结果不准确。"
				)
			)
			with gr.Row():
				some_input_audio = gr.File(label=i18n("上传音频"), type="filepath")
				with gr.Column():
					audio_bpm = gr.Number(label=i18n("输入音频BPM"), value=120, interactive=True)
					some_output_folder = gr.Textbox(label=i18n("输出目录"), value=webui_config["tools"]["store_dir"] if webui_config["tools"]["store_dir"] else "results/", interactive=True, scale=3)
					with gr.Row():
						select_some_output_dir = gr.Button(i18n("选择文件夹"))
						open_some_output_dir = gr.Button(i18n("打开文件夹"))
			some_button = gr.Button(i18n("开始转换"), variant="primary")
			output_message_some = gr.Textbox(label="Output Message")
			gr.Markdown(i18n("### 注意事项"))
			gr.Markdown(
				i18n(
					"1. 音频BPM (每分钟节拍数) 可以通过MixMeister BPM Analyzer等软件测量获取。<br>2. 为保证MIDI提取质量, 音频文件请采用干净清晰无混响底噪人声。<br>3. 输出MIDI不带歌词信息, 需要用户自行添加歌词。<br>4. 实际使用体验中部分音符会出现断开的现象, 需自行修正。SOME的模型主要面向DiffSinger唱法模型自动标注, 比正常用户在创作中需要的MIDI更加精细, 因而可能导致模型倾向于对音符进行切分。<br>5. 提取的MIDI没有量化/没有对齐节拍/不适配BPM, 需自行到各编辑器中手动调整。"
				)
			)

	convert_audio_button.click(
		fn=convert_audio, inputs=[inputs, ffmpeg_output_format, ffmpeg_output_folder, sample_rate, channels, wav_bit_depth, flac_bit_depth, mp3_bit_rate, ogg_bit_rate], outputs=output_message_ffmpeg
	)
	merge_audio_button.click(fn=merge_audios, inputs=[merge_audio_input, merge_audio_output], outputs=output_message_merge)
	compute_sdr_button.click(fn=caculate_sdr, inputs=[reference_audio, estimated_audio], outputs=output_message_sdr)
	some_button.click(fn=some_inference, inputs=[some_input_audio, audio_bpm, some_output_folder], outputs=output_message_some)
	select_ffmpeg_output_dir.click(fn=select_folder, outputs=ffmpeg_output_folder)
	open_ffmpeg_output_dir.click(fn=open_folder, inputs=ffmpeg_output_folder)
	select_merge_input_dir.click(fn=select_folder, outputs=merge_audio_input)
	open_merge_input_dir.click(fn=open_folder, inputs=merge_audio_input)
	select_merge_output_dir.click(fn=select_folder, outputs=merge_audio_output)
	open_merge_output_dir.click(fn=open_folder, inputs=merge_audio_output)
	select_some_output_dir.click(fn=select_folder, outputs=some_output_folder)
	open_some_output_dir.click(fn=open_folder, inputs=some_output_folder)
