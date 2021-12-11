import os
import glob
import json
import numpy as np
import argparse


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Check telegram image number.", allow_abbrev=False)
parser.add_argument("-t",
                    "--telegram_image_dir",
                    help="Path to the folder where user uploaded image from Telegram is stored permanently.",
                    type=str,
                    default=None, required=True)

args = parser.parse_args()

def main():

    telegram_img_dir = args.telegram_image_dir + 'telegram_img/'
    number = []
    for path in glob.glob(telegram_img_dir + '*.jpg'):
        file_num = int(path.split("\\")[-1].split("_")[-1].split(".")[0])
        number.append(file_num)
    if bool(number):
        number.sort()
    else:
        number = [0]
    my_number = np.arange(1, number[-1]+1, 1, dtype='uint').tolist()
    filtered_number = list(filter(lambda x: x not in number, my_number))

    if bool(filtered_number):
        start_number = filtered_number
        start_number.append(number[-1] + 1)
    else:
        start_number = [number[-1] + 1]

    dumped_json = json.dumps(start_number)
    loaded_json = json.loads(dumped_json)
    print(dumped_json)

    return start_number

if __name__=='__main__':
    main()






