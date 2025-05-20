__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import os
import gradio as gr
import zipfile
import shutil
from webui.utils import i18n, logger
from tqdm import tqdm


def reflash_files():
	file_lists = ""
	if os.path.exists("input"):
		file_lists += i18n("input文件夹内文件列表:\n")
		for file in os.listdir("input"):
			file_lists += file + "  "
	else:
		file_lists += i18n("input文件夹为空\n")
	file_lists += "\n\n"
	if os.path.exists("results"):
		file_lists += i18n("results文件夹内文件列表:\n")
		for file in os.listdir("results"):
			file_lists += file + "  "
	else:
		file_lists += i18n("results文件夹为空\n")
	return file_lists


def delete_input_files():
	if os.path.exists("input"):
		shutil.rmtree("input")
	os.makedirs("input")
	gr.Info(i18n("已删除input文件夹内所有文件"))
	logger.info("Successfully deleted all files in the input folder.")
	return reflash_files()


def delete_results_files():
	if os.path.exists("results"):
		shutil.rmtree("results")
	os.makedirs("results")
	gr.Info(i18n("已删除results文件夹内所有文件"))
	logger.info("Successfully deleted all files in the results folder.")
	return reflash_files()


def download_results_files():
	if os.path.exists("results"):
		if os.path.exists("results.zip"):
			os.remove("results.zip")
		logger.info("Successfully deleted the old zip file.")
		with zipfile.ZipFile("results.zip", "w") as zipf:
			for dir, _, files in os.walk("results"):
				for file in tqdm(files, desc="Creating zip file"):
					zipf.write(os.path.join(dir, file), os.path.relpath(os.path.join(dir, file), "results"))
					logger.debug(f"Added {file} to the zip file.")
		logger.info("Successfully created the zip file results.zip.")
		return "results.zip"
	return None


def upload_files_to_input(upload_files, auto_unzip):
	for file in upload_files:
		if file.name.endswith(".zip") and auto_unzip:
			with zipfile.ZipFile(file, "r") as zip_ref:
				zip_ref.extractall("input")
		else:
			try:
				shutil.copy(file, "input")
			except shutil.SameFileError:
				logger.warning(f"File {file} already exists in the input folder. Skipping...")
	return reflash_files()
