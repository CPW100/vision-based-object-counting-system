import os
import glob
import time
import shutil
import argparse
import pandas as pd
from datetime import datetime

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Check model path.", allow_abbrev=False)
parser.add_argument("-n",
                    "--node_red_directory",
                    help="Path to the node-red directory in streamlit user directory.",
                    type=str,
                    default=None, required=True)

args = parser.parse_args()

r"""
To copy image file from 'temp_img_directory' to
'telegram_img_directory' and clear out the files
in 'temp_img_directory'.
-> May need to resize the image in the future
"""
def main():
    temp_img_dir = args.node_red_directory + 'temp_img'
    telegram_img_dir = args.node_red_directory + 'telegram_img'
    temp_img_list = [temp_img_dir + "/" + file for file in  os.listdir(temp_img_dir)]
    for img_file in os.listdir(temp_img_dir):
        try:
            src = temp_img_dir + "/" + img_file
            dst = telegram_img_dir + "/" + img_file
            if os.path.isfile(src):
                shutil.copy(src, dst)
        except Exception as e:
            print('Failed to copy %s to %s. Reason: %s' % (src, dst, e))
        try:
            if os.path.isfile(src):
                os.remove(src)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (src, e))


if __name__ == "__main__":
    now = datetime.now()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Executing script_5.py\n")
    main()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Completed execution of script_5.py\n")



