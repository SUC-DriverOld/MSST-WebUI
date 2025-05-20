import os
import shutil
import qfluentwidgets
import argparse


def fix_JP():
	package_path = os.path.dirname(qfluentwidgets.__file__)
	target_path = os.path.join(package_path, "_rc", "resource.py")
	if not os.path.exists(target_path):
		raise FileNotFoundError("target file not found!")
	os.rename(target_path, target_path + ".backup")

	new_resource_pak = os.path.join(os.path.dirname(__file__), "resource", "i18n", "resource.py")
	if not os.path.exists(new_resource_pak):
		raise FileNotFoundError("new resource file not found!")
	shutil.copyfile(new_resource_pak, target_path)
	print("Done!")


def restore():
	package_path = os.path.dirname(qfluentwidgets.__file__)
	target_path = os.path.join(package_path, "_rc", "resource.py")
	if not os.path.exists(target_path + ".backup"):
		raise FileNotFoundError("backup file not found!")
	os.remove(target_path)
	os.rename(target_path + ".backup", target_path)
	print("Done!")


def main():
	parser = argparse.ArgumentParser(description="Fix or restore the resource file.")
	parser.add_argument("-a", "--action", choices=["fix", "restore"], help="Action to perform: 'fix' to update the resource file or 'restore' to restore from backup")
	args = parser.parse_args()

	if args.action == "fix":
		fix_JP()
	elif args.action == "restore":
		restore()


if __name__ == "__main__":
	main()
