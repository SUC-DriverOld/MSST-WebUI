from PySide6.QtCore import QThread, Signal
import os
from time import sleep
import time
import requests


class DownloadThread(QThread):
	update_single_progress = Signal(int)  # 更新单个文件进度
	update_total_progress = Signal(int)  # 更新总进度
	update_speed = Signal(str)  # 更新下载速度

	def __init__(self, urls, target_dir):
		super().__init__()
		self.urls = urls
		self.target_dir = target_dir
		self.total_progress = 0
		self.total_files = len(urls)
		self.total_downloaded = 0
		self.download_speed = 0

	def run(self):
		"""执行下载任务"""
		for url in self.urls:
			self.download_model(url)

	def download_model(self, url):
		model_filename = url.split("/")[-1]
		category = url.split("/")[-2]
		os.makedirs(os.path.join(self.target_dir, category), exist_ok=True)
		file_path = os.path.join(self.target_dir, category, model_filename)

		response = requests.get(url, stream=True)
		total_size = int(response.headers.get("Content-Length", 0))

		downloaded = 0
		start_time = time.time()
		last_downloaded = 0

		with open(file_path, "wb") as f:
			for chunk in response.iter_content(chunk_size=1024):
				if chunk:
					f.write(chunk)
					downloaded += len(chunk)

					self.update_single_progress.emit(int(downloaded / total_size * 100))

					elapsed_time = time.time() - start_time
					if elapsed_time > 1:
						self.download_speed = (downloaded - last_downloaded) / elapsed_time
						self.update_speed.emit(f"{self.download_speed / 1024 / 1024:.2f} MB/s")
						start_time = time.time()
						last_downloaded = downloaded

		self.update_speed.emit("准备下载...")
		self.total_progress += 1
		self.update_total_progress.emit(int(self.total_progress / self.total_files * 100))
