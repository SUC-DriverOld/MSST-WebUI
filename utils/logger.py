import logging
import os
from colorama import Fore, Style
from datetime import datetime
from colorama import init

init(autoreset=True)

MAX_LOG = 100
LOG_DIR = "logs"
LOG_FILENAME_ENV = "MSST_LOG_FILE"


class ColorFormatter(logging.Formatter):
	LEVEL_STYLES = {
		"INFO": f"[{Fore.GREEN}INFO{Style.RESET_ALL}]    ",
		"DEBUG": f"[{Fore.BLUE}DEBUG{Style.RESET_ALL}]   ",
		"WARNING": f"[{Fore.YELLOW}WARNING{Style.RESET_ALL}] ",
		"ERROR": f"[{Fore.RED}ERROR{Style.RESET_ALL}]   ",
		"CRITICAL": f"[{Fore.MAGENTA}CRITICAL{Style.RESET_ALL}]",
	}

	def format(self, record):
		record.pathname = os.path.relpath(record.pathname)
		log_msg = super().format(record)
		if record.levelname in self.LEVEL_STYLES:
			log_msg = log_msg.replace(record.levelname, self.LEVEL_STYLES[record.levelname])
		return log_msg


def manage_log_files(log_dir, max_log):
	def parse_date(filename):
		for fmt in ("%Y-%m-%d", "%Y-%m-%d_%H-%M-%S"):
			try:
				return datetime.strptime(filename.split(".")[0], fmt)
			except ValueError:
				continue
		return datetime.min

	log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
	log_files = sorted(log_files, key=parse_date)
	while len(log_files) > max_log:
		try:
			oldest_file = log_files.pop(0)
			os.remove(os.path.join(log_dir, oldest_file))
		except Exception as e:
			pass


def set_log_level(logger, level):
	logger.console_handler.setLevel(level)


def get_logger(console_level=logging.INFO, max_log=MAX_LOG):
	logger = logging.getLogger("logger")
	if logger.hasHandlers():
		return logger

	logger.setLevel(logging.DEBUG)
	console_handler = logging.StreamHandler()
	console_handler.setLevel(console_level)
	formatter = ColorFormatter(fmt="%(asctime)s.%(msecs)03d %(levelname)s[%(pathname)s:%(lineno)d] %(message)s", datefmt="%H:%M:%S")
	console_handler.setFormatter(formatter)

	os.makedirs(LOG_DIR, exist_ok=True)
	log_filename = os.environ.get(LOG_FILENAME_ENV, None)
	if not log_filename:
		log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
		os.environ[LOG_FILENAME_ENV] = log_filename
		manage_log_files(LOG_DIR, max_log)
	file_path = os.path.join(LOG_DIR, log_filename)

	file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
	file_handler.setLevel(logging.DEBUG)
	file_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
	file_handler.setFormatter(file_formatter)

	if not logger.hasHandlers():
		logger.addHandler(console_handler)
		logger.addHandler(file_handler)

	logger.console_handler = console_handler
	logger.file_handler = file_handler

	return logger


if __name__ == "__main__":
	logger = get_logger(console_level=logging.INFO)
	logger.debug("This is a debug message.")
	logger.info("This is an info message.")
	logger.warning("This is a warning message.")
	logger.error("This is an error message.")
	logger.critical("This is a critical message.")
	set_log_level(logger, logging.DEBUG)
	logger.debug("This is a debug message after log level change.")
	logger.info("This is an info message after log level change.")
