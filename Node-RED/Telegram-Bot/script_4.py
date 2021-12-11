import os
import time
import shutil
import argparse
from datetime import datetime


parser = argparse.ArgumentParser(
    description="Check telegram image number.", allow_abbrev=False)
parser.add_argument("-n",
                    "--temp_image_directory",
                    help="Temporary telegram image directory.",
                    type=str,
                    default=None, required=True)

args = parser.parse_args()

def main():
    folder = args.temp_image_directory
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__=='__main__':
    now = datetime.now()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Executing script_4.py\n")
    main()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Completed execution of script_4.py\n")




