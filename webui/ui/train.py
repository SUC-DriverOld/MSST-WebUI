__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gradio as gr
import pandas as pd

from multiprocessing import cpu_count
from utils.constant import *
from webui.utils import i18n, open_folder, select_file, select_folder, select_yaml_file
from webui.train import save_training_config, start_training, update_train_start_check_point, validate_model, load_augmentations_config, stop_msst_valid


def train(webui_config, device):
	device = [value for _, value in device.items()]

	gr.Markdown(
		value=i18n(
			"此页面提供数据集制作教程, 训练参数选择, 以及一键训练。有关配置文件的修改和数据集文件夹的详细说明请参考MSST原项目: [https://github.com/ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)<br>在开始下方的模型训练之前, 请先进行训练数据的制作。<br>说明: 数据集类型即训练集制作Step 1中你选择的类型, 1: Type1; 2: Type2; 3: Type3; 4: Type4, 必须与你的数据集类型相匹配。"
		)
	)
	with gr.Tabs():
		with gr.TabItem(label=i18n("训练")):
			with gr.Row():
				train_model_type = gr.Dropdown(
					label=i18n("选择训练模型类型"), choices=MODEL_TYPE, value=webui_config["training"]["model_type"] if webui_config["training"]["model_type"] else None, interactive=True, scale=1
				)
				train_config_path = gr.Textbox(
					label=i18n("配置文件路径"),
					value=webui_config["training"]["config_path"] if webui_config["training"]["config_path"] else i18n("请输入配置文件路径或选择配置文件"),
					interactive=True,
					scale=3,
				)
				select_train_config_path = gr.Button(i18n("选择配置文件"), scale=1)
			with gr.Row():
				train_dataset_type = gr.Radio(
					label=i18n("数据集类型"), choices=[1, 2, 3, 4], value=webui_config["training"]["dataset_type"] if webui_config["training"]["dataset_type"] else None, interactive=True, scale=1
				)
				train_dataset_path = gr.Textbox(
					label=i18n("数据集路径"),
					value=webui_config["training"]["dataset_path"] if webui_config["training"]["dataset_path"] else i18n("请输入或选择数据集文件夹"),
					interactive=True,
					scale=3,
				)
				select_train_dataset_path = gr.Button(i18n("选择数据集文件夹"), scale=1)
			with gr.Row():
				train_valid_path = gr.Textbox(
					label=i18n("验证集路径"), value=webui_config["training"]["valid_path"] if webui_config["training"]["valid_path"] else i18n("请输入或选择验证集文件夹"), interactive=True, scale=4
				)
				select_train_valid_path = gr.Button(i18n("选择验证集文件夹"), scale=1)
			with gr.Row():
				train_results_path = gr.Textbox(
					label=i18n("模型保存路径"),
					value=webui_config["training"]["results_path"] if webui_config["training"]["results_path"] else i18n("请输入或选择模型保存文件夹"),
					interactive=True,
					scale=3,
				)
				select_train_results_path = gr.Button(i18n("选择文件夹"), scale=1)
				open_train_results_path = gr.Button(i18n("打开文件夹"), scale=1)
			with gr.Row():
				train_start_check_point = gr.Dropdown(label=i18n("选择初始模型, 若无初始模型, 留空或选择None即可"), choices=["None"], value="None", interactive=True, scale=4)
				reflesh_start_check_point = gr.Button(i18n("刷新初始模型列表"), scale=1)
			with gr.Accordion(i18n("训练参数设置"), open=True):
				with gr.Row():
					train_device_ids = gr.CheckboxGroup(
						label=i18n("选择使用的GPU"), choices=device, value=webui_config["training"]["device"] if webui_config["training"]["device"] else device[0], interactive=True
					)
					train_num_workers = gr.Number(
						label=i18n("num_workers: 数据集读取线程数, 0为自动"),
						value=webui_config["training"]["num_workers"] if webui_config["training"]["num_workers"] else 0,
						interactive=True,
						minimum=0,
						maximum=cpu_count(),
						step=1,
					)
					train_seed = gr.Number(label=i18n("随机数种子, 0为随机"), value=0)
				with gr.Row():
					train_pin_memory = gr.Checkbox(label=i18n("是否将加载的数据放置在固定内存中, 默认为否"), value=webui_config["training"]["pin_memory"], interactive=True)
					train_accelerate = gr.Checkbox(label=i18n("是否使用加速训练, 对于多显卡用户会加快训练"), value=webui_config["training"]["accelerate"], interactive=True)
					train_pre_validate = gr.Checkbox(label=i18n("是否在训练前验证模型, 默认为否"), value=webui_config["training"]["pre_valid"], interactive=True)
				with gr.Row():
					train_use_multistft_loss = gr.Checkbox(label=i18n("是否使用MultiSTFT Loss, 默认为否"), value=webui_config["training"]["use_multistft_loss"], interactive=True)
					train_use_mse_loss = gr.Checkbox(label=i18n("是否使用MSE loss, 默认为否"), value=webui_config["training"]["use_mse_loss"], interactive=True)
					train_use_l1_loss = gr.Checkbox(label=i18n("是否使用L1 loss, 默认为否"), value=webui_config["training"]["use_l1_loss"], interactive=True)
				with gr.Row():
					train_metrics_list = gr.CheckboxGroup(
						label=i18n("选择输出的评估指标"), choices=METRICS, value=webui_config["training"]["metrics"] if webui_config["training"]["metrics"] else METRICS[0], interactive=True
					)
					train_metrics_scheduler = gr.Radio(
						label=i18n("选择调度器使用的评估指标"),
						choices=METRICS,
						value=webui_config["training"]["metrics_scheduler"] if webui_config["training"]["metrics_scheduler"] else METRICS[0],
						interactive=True,
					)
			save_train_config = gr.Button(i18n("保存上述训练配置"))
			start_train_button = gr.Button(i18n("开始训练"), variant="primary")
			gr.Markdown(value=i18n("点击开始训练后, 请到终端查看训练进度或报错, 下方不会输出报错信息, 想要停止训练可以直接关闭终端。在训练过程中, 你也可以关闭网页, 仅**保留终端**。"))
			output_message_train = gr.Textbox(label="Output Message")

			select_train_config_path.click(fn=select_yaml_file, outputs=train_config_path)
			select_train_dataset_path.click(fn=select_folder, outputs=train_dataset_path)
			select_train_valid_path.click(fn=select_folder, outputs=train_valid_path)
			select_train_results_path.click(fn=select_folder, outputs=train_results_path)
			open_train_results_path.click(fn=open_folder, inputs=train_results_path)
			reflesh_start_check_point.click(fn=update_train_start_check_point, inputs=train_results_path, outputs=train_start_check_point)
			save_train_config.click(
				fn=save_training_config,
				inputs=[
					train_model_type,
					train_config_path,
					train_dataset_type,
					train_dataset_path,
					train_valid_path,
					train_num_workers,
					train_device_ids,
					train_seed,
					train_pin_memory,
					train_use_multistft_loss,
					train_use_mse_loss,
					train_use_l1_loss,
					train_results_path,
					train_accelerate,
					train_pre_validate,
					train_metrics_list,
					train_metrics_scheduler,
				],
				outputs=output_message_train,
			)
			start_train_button.click(
				fn=start_training,
				inputs=[
					train_model_type,
					train_config_path,
					train_dataset_type,
					train_dataset_path,
					train_valid_path,
					train_num_workers,
					train_device_ids,
					train_seed,
					train_pin_memory,
					train_use_multistft_loss,
					train_use_mse_loss,
					train_use_l1_loss,
					train_results_path,
					train_start_check_point,
					train_accelerate,
					train_pre_validate,
					train_metrics_list,
					train_metrics_scheduler,
				],
				outputs=output_message_train,
			)

		with gr.TabItem(label=i18n("验证")):
			gr.Markdown(
				value=i18n(
					"此页面用于手动验证模型效果, 测试验证集, 输出SDR测试信息。输出的信息会存放在输出文件夹的results.txt中。<br>下方参数将自动加载训练页面的参数, 在训练页面点击保存训练参数后, 重启WebUI即可自动加载。当然你也可以手动输入参数。<br>"
				)
			)
			with gr.Row():
				valid_model_type = gr.Dropdown(
					label=i18n("选择模型类型"), choices=MODEL_TYPE, value=webui_config["training"]["model_type"] if webui_config["training"]["model_type"] else None, interactive=True, scale=1
				)
				valid_config_path = gr.Textbox(
					label=i18n("配置文件路径"),
					value=webui_config["training"]["config_path"] if webui_config["training"]["config_path"] else i18n("请输入配置文件路径或选择配置文件"),
					interactive=True,
					scale=3,
				)
				select_valid_config_path = gr.Button(i18n("选择配置文件"), scale=1)
			with gr.Row():
				valid_model_path = gr.Textbox(label=i18n("模型路径"), value=i18n("请输入或选择模型文件"), interactive=True, scale=4)
				select_valid_model_path = gr.Button(i18n("选择模型文件"), scale=1)
			with gr.Row():
				valid_path = gr.Textbox(
					label=i18n("验证集路径"), value=webui_config["training"]["valid_path"] if webui_config["training"]["valid_path"] else i18n("请输入或选择验证集文件夹"), interactive=True, scale=4
				)
				select_valid_path = gr.Button(i18n("选择验证集文件夹"), scale=1)
			with gr.Row():
				valid_results_path = gr.Textbox(label=i18n("输出目录"), value="results/", interactive=True, scale=3)
				select_valid_results_path = gr.Button(i18n("选择文件夹"), scale=1)
				open_valid_results_path = gr.Button(i18n("打开文件夹"), scale=1)
			with gr.Accordion(i18n("验证参数设置"), open=True):
				with gr.Row():
					valid_device_ids = gr.CheckboxGroup(
						label=i18n("选择使用的GPU"), choices=device, value=webui_config["training"]["device"] if webui_config["training"]["device"] else device[0], interactive=True
					)
					valid_extension = gr.Radio(label=i18n("选择验证集音频格式"), choices=["wav", "flac", "mp3"], value="wav", interactive=True)
					valid_num_workers = gr.Number(
						label=i18n("验证集读取线程数, 0为自动"),
						value=webui_config["training"]["num_workers"] if webui_config["training"]["num_workers"] else 0,
						interactive=True,
						minimum=0,
						maximum=cpu_count(),
						step=1,
					)

				with gr.Row():
					with gr.Column():
						vaild_metrics = gr.CheckboxGroup(
							label=i18n("选择输出的评估指标"), choices=METRICS, value=webui_config["training"]["metrics"] if webui_config["training"]["metrics"] else METRICS[0], interactive=True
						)
					with gr.Column():
						valid_pin_memory = gr.Checkbox(label=i18n("是否将加载的数据放置在固定内存中, 默认为否"), value=webui_config["training"]["pin_memory"], interactive=True)
						valid_use_tta = gr.Checkbox(label=i18n("启用TTA, 能小幅提高分离质量, 若使用, 推理时间x3"), value=False, interactive=True)
			valid_button = gr.Button(i18n("开始验证"), variant="primary")
			with gr.Row():
				valid_output_message = gr.Textbox(label="Output Message", scale=4)
				stop_valid = gr.Button(i18n("强制停止"), scale=1)

			select_valid_config_path.click(fn=select_yaml_file, outputs=valid_config_path)
			select_valid_model_path.click(fn=select_file, outputs=valid_model_path)
			select_valid_path.click(fn=select_folder, outputs=valid_path)
			select_valid_results_path.click(fn=select_folder, outputs=valid_results_path)
			open_valid_results_path.click(fn=open_folder, inputs=valid_results_path)
			valid_button.click(
				fn=validate_model,
				inputs=[
					valid_model_type,
					valid_config_path,
					valid_model_path,
					valid_path,
					valid_results_path,
					valid_device_ids,
					valid_num_workers,
					valid_extension,
					valid_pin_memory,
					valid_use_tta,
					vaild_metrics,
				],
				outputs=valid_output_message,
			)
			stop_valid.click(fn=stop_msst_valid)

		with gr.TabItem(label=i18n("训练集制作指南")):
			with gr.Accordion(i18n("Step 1: 数据集制作"), open=False):
				gr.Markdown(
					value=i18n("请**任选下面四种类型之一**制作数据集文件夹, 并按照给出的目录层级放置你的训练数据。完成后, 记录你的数据集**文件夹路径**以及你选择的**数据集类型**, 以便后续使用。")
				)
				with gr.Row():
					with gr.Column():
						gr.Markdown("# Type 1 (MUSDB)")
						gr.Markdown(i18n("不同的文件夹。每个文件夹包含所需的所有stems, 格式为stem_name.wav。与MUSDBHQ18数据集相同。在最新的代码版本中, 可以使用flac替代wav。<br>例如: "))
						gr.Markdown(
							"""your_datasets_folder<br>├───Song 1<br>│   ├───vocals.wav<br>│   ├───bass.wav<br>│   ├───drums.wav<br>│   └───other.wav<br>├───Song 2<br>│   ├───vocals.wav<br>│   ├───bass.wav<br>│   ├───drums.wav<br>│   └───other.wav<br>├───Song 3<br>    └───...<br>"""
						)
					with gr.Column():
						gr.Markdown("# Type 2 (Stems)")
						gr.Markdown(i18n("每个文件夹是stem_name。文件夹中包含仅由所需stem组成的wav文件。<br>例如: "))
						gr.Markdown(
							"""your_datasets_folder<br>├───vocals<br>│   ├───vocals_1.wav<br>│   ├───vocals_2.wav<br>│   ├───vocals_3.wav<br>│   └───...<br>├───bass<br>│   ├───bass_1.wav<br>│   ├───bass_2.wav<br>│   ├───bass_3.wav<br>│   └───...<br>├───drums<br>    └───...<br>"""
						)
					with gr.Column():
						gr.Markdown("# Type 3 (CSV file)")
						gr.Markdown(i18n("可以提供以下结构的CSV文件 (或CSV文件列表) <br>例如: "))
						gr.Markdown(
							"""instrum,path<br>vocals,/path/to/dataset/vocals_1.wav<br>vocals,/path/to/dataset2/vocals_v2.wav<br>vocals,/path/to/dataset3/vocals_some.wav<br>...<br>drums,/path/to/dataset/drums_good.wav<br>...<br>"""
						)
					with gr.Column():
						gr.Markdown("# Type 4 (MUSDB Aligned)")
						gr.Markdown(i18n("与类型1相同, 但在训练过程中所有乐器都将来自歌曲的相同位置。<br>例如: "))
						gr.Markdown(
							"""your_datasets_folder<br>├───Song 1<br>│   ├───vocals.wav<br>│   ├───bass.wav<br>│   ├───drums.wav<br>│   └───other.wav<br>├───Song 2<br>│   ├───vocals.wav<br>│   ├───bass.wav<br>│   ├───drums.wav<br>│   └───other.wav<br>├───Song 3<br>    └───...<br>"""
						)
			with gr.Accordion(i18n("Step 2: 验证集制作"), open=False):
				gr.Markdown(
					value=i18n(
						"验证集制作。验证数据集**必须**与上面数据集制作的Type 1(MUSDB)数据集**结构相同** (**无论你使用哪种类型的数据集进行训练**) , 此外每个文件夹还必须包含每首歌的mixture.wav, mixture.wav是所有stem的总和<br>例如: "
					)
				)
				gr.Markdown(
					"""your_datasets_folder<br>├───Song 1<br>│   ├───vocals.wav<br>│   ├───bass.wav<br>│   ├───drums.wav<br>│   ├───other.wav<br>│   └───mixture.wav<br>├───Song 2<br>│   ├───vocals.wav<br>│   ├───bass.wav<br>│   ├───drums.wav<br>│   ├───other.wav<br>│   └───mixture.wav<br>├───Song 3<br>    └───...<br>"""
				)
			with gr.Accordion(i18n("Step 3: 选择并修改修改配置文件"), open=False):
				gr.Markdown(
					value=i18n("请先明确你想要训练的模型类型, 然后选择对应的配置文件进行修改。<br>目前有以下几种模型类型: ")
					+ str(MODEL_TYPE)
					+ i18n(
						"<br>确定好模型类型后, 你可以前往整合包根目录中的configs_backup文件夹下找到对应的配置文件模板。复制一份模板, 然后根据你的需求进行修改。修改完成后记下你的配置文件路径, 以便后续使用。<br>特别说明: config_musdb18_xxx.yaml是针对MUSDB18数据集的配置文件。<br>"
					)
				)
				open_config_template = gr.Button(i18n("打开配置文件模板文件夹"), variant="primary")
				gr.Markdown(value=i18n("你可以使用下表根据你的GPU选择用于训练的BS_Roformer模型的batch_size参数。表中提供的批量大小值适用于单个GPU。如果你有多个GPU, 则需要将该值乘以GPU的数量。"))
				roformer_data = {
					"chunk_size": [131584, 131584, 131584, 131584, 131584, 131584, 263168, 263168, 352800, 352800, 352800, 352800],
					"dim": [128, 256, 384, 512, 256, 256, 128, 256, 128, 256, 384, 512],
					"depth": [6, 6, 6, 6, 8, 12, 6, 6, 6, 6, 12, 12],
					"batch_size (A6000 48GB)": [10, 8, 7, 6, 6, 4, 4, 3, 2, 2, 1, "-"],
					"batch_size (3090/4090 24GB)": [5, 4, 3, 3, 3, 2, 2, 1, 1, 1, "-", "-"],
					"batch_size (16GB)": [3, 2, 2, 2, 2, 1, 1, 1, "-", "-", "-", "-"],
				}
				gr.DataFrame(pd.DataFrame(roformer_data))
			with gr.Accordion(i18n("Step 4: 数据增强"), open=False):
				gr.Markdown(
					value=i18n(
						"数据增强可以动态更改stem, 通过从旧样本创建新样本来增加数据集的大小。现在, 数据增强的控制在配置文件中进行。下面是一个包含所有可用数据增强的完整配置示例。你可以将其复制到你的配置文件中以使用数据增强。<br>注意:<br>1. 要完全禁用所有数据增强, 可以从配置文件中删除augmentations部分或将enable设置为false。<br>2. 如果要禁用某些数据增强, 只需将其设置为0。<br>3. all部分中的数据增强应用于所有stem。<br>4. vocals/bass等部分中的数据增强仅应用于相应的stem。你可以为training.instruments中给出的所有stem创建这样的部分。"
					)
				)
				gr.Code(value=load_augmentations_config(), language="yaml")

			open_config_template.click(open_folder, inputs=gr.Textbox("configs_backup", visible=False))
