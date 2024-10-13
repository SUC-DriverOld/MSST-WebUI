import os
import subprocess
import numpy as np
import librosa
from pydub import AudioSegment

from utils.constant import *
from tools.webUI.utils import i18n, load_configs, save_configs, print_command

def convert_audio(uploaded_files, ffmpeg_output_format, ffmpeg_output_folder):
    if not uploaded_files:
        return i18n("请上传至少一个文件")

    success_files = []

    for uploaded_file in uploaded_files:
        uploaded_file_path = uploaded_file.name
        output_path = ffmpeg_output_folder

        os.makedirs(output_path, exist_ok=True)

        config = load_configs(WEBUI_CONFIG)
        config['tools']['ffmpeg_output_format'] = ffmpeg_output_format
        config['tools']['ffmpeg_output_folder'] = ffmpeg_output_folder
        save_configs(config, WEBUI_CONFIG)

        output_file = os.path.join(output_path, os.path.splitext(os.path.basename(uploaded_file_path))[0] + "." + ffmpeg_output_format)
        command = f"{FFMPEG} -i \"{uploaded_file_path}\" \"{output_file}\""
        print_command(command)

        try:
            subprocess.run(command, shell=True, check=True)
            success_files.append(output_file)
        except subprocess.CalledProcessError:
            print(f"Fail to convert file: {uploaded_file_path}\n")
            continue

    if not success_files:
        return i18n("所有文件转换失败, 请检查文件格式和ffmpeg路径。")

    else:
        text = i18n("处理完成, 文件已保存为: ") + "\n" + "\n".join(success_files)
        return text

def merge_audios(input_folder, output_folder):
    config = load_configs(WEBUI_CONFIG)
    config['tools']['merge_audio_input'] = input_folder
    config['tools']['merge_audio_output'] = output_folder
    save_configs(config, WEBUI_CONFIG)

    combined_audio = AudioSegment.empty()
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"merged_audio_{os.path.basename(input_folder)}.wav")

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):
            file_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_file(file_path)
            combined_audio += audio

    try:
        combined_audio.export(output_file, format="wav")
        return i18n("处理完成, 文件已保存为: ") + output_file
    except Exception as e:
        print(e)
        return i18n("处理失败!")

def process_audio(reference_path, estimated_path):
    reference, _ = librosa.load(reference_path, sr=44100, mono=False)

    if reference.ndim == 1:
        reference = np.vstack((reference, reference))

    estimated, _ = librosa.load(estimated_path, sr=44100, mono=False)

    if estimated.ndim == 1:
        estimated = np.vstack((estimated, estimated))

    min_length = min(reference.shape[1], estimated.shape[1])
    reference = reference[:, :min_length]
    estimated = estimated[:, :min_length]

    sdr_values = []

    for i in range(reference.shape[0]):
        num = np.sum(np.square(reference[i])) + 1e-7
        den = np.sum(np.square(reference[i] - estimated[i])) + 1e-7
        sdr_values.append(round(10 * np.log10(num / den), 4))

    average_sdr = np.mean(sdr_values)

    print(f"[INFO] SDR Values: {sdr_values}, Average SDR: {average_sdr:.4f}")
    return f"SDR Values: {sdr_values}, Average SDR: {average_sdr:.4f}"

def ensemble(files, ensemble_mode, weights, output_path):
    if len(files) < 2:
        return i18n("请上传至少2个文件")

    if len(files) != len(weights.split()):
        return i18n("上传的文件数目与权重数目不匹配")
    else:
        config = load_configs(WEBUI_CONFIG)
        config['tools']['ensemble_type'] = ensemble_mode
        config['tools']['ensemble_output_folder'] = output_path
        save_configs(config, WEBUI_CONFIG)

        files = [f"\"{file}\"" for file in files]
        file_basename = [os.path.splitext(os.path.basename(file))[0] for file in files]
        files_argument = " ".join(files)

        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"ensemble_{ensemble_mode}_{'_'.join(file_basename)}.wav")
        command = f"{PYTHON} utils/ensemble.py --files {files_argument} --type {ensemble_mode} --weights {weights} --output {output_path}"
        print_command(command)

        try:
            subprocess.run(command, shell = True)
            return i18n("处理完成, 文件已保存为: ") + output_path
        except Exception as e:
            return i18n("处理失败!")

def some_inference(audio_file, bpm, output_dir):
    if not os.path.isfile(SOME_WEIGHT):
        return i18n("请先下载SOME预处理模型并放置在tools/SOME_weights文件夹下! ")
    if not audio_file.endswith('.wav'):
        return i18n("请上传wav格式文件")

    os.makedirs(output_dir, exist_ok=True)

    config = load_configs(WEBUI_CONFIG)
    config['tools']['some_output_folder'] = output_dir
    save_configs(config, WEBUI_CONFIG)

    tempo = int(bpm)
    file_name = os.path.basename(audio_file)[0:-4]
    midi = os.path.join(output_dir, f"{file_name}.mid")
    command = f"{PYTHON} tools/SOME/infer.py --model {SOME_WEIGHT} --wav \"{audio_file}\" --midi \"{midi}\" --tempo {tempo}"
    print_command(command)

    try:
        subprocess.run(command, shell=True)
        return i18n("处理完成, 文件已保存为: ") + midi
    except Exception as e:
        return i18n("处理失败!")