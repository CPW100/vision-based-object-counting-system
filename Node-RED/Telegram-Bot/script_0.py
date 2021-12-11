import os
import glob
import math
import shutil 
import argparse
import pandas as pd
from datetime import datetime

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Check model path.", allow_abbrev=False)
parser.add_argument("-s",
                    "--streamlit_user_dir_fpath",
                    help="Path to the folder where Streamlit user directory is stored.",
                    type=str,
                    default=None, required=True)

args = parser.parse_args()
x_folder_subdir = ("fine_tuned", "node-red", "dataset")
root_dir = args.streamlit_user_dir_fpath.split('/')
root_dir = '/'.join(root_dir)


def create_dir_1(path):
    for i in range(len(x_folder_subdir)):
        fpath = path + "/" + x_folder_subdir[i]
        if not os.path.isdir(fpath):
            os.mkdir(fpath)


def create_dir_2(user_dir, folder_name):

    sub_folder_list = ['telegram_img', 'csv_prediction_log', 'bbox_img', 'temp_img']
    x = x_folder_subdir[1]

    root_path = user_dir + "/" + x + "/" + folder_name
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    for i in range(len(sub_folder_list)):
        abs_path = root_path + "/" + sub_folder_list[i]
        if not os.path.isdir(abs_path):
            os.mkdir(abs_path)

def create_dir_3(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def main():


    # Create folder directory
    user_dir_list = [root_dir + "/" + item for item in os.listdir(root_dir)]

    # Perform copy file operation
    temp_list = root_dir.split('/')
    key_1 = temp_list.pop()
    main_dir = "/".join(temp_list)
    server_dir = main_dir + "/" + "user_server"
    csv_dir = main_dir + "/" + 'user_csv'
    create_dir_3(server_dir)
    create_dir_3(csv_dir)

    count = 0
    port = 8600
    config_dict = []
    count_b = 0
    for user_dir in user_dir_list:
        create_dir_1(user_dir)
        usr_dir_name = user_dir.split("/")[-1]
        dir_server_dataset = server_dir + "/" + usr_dir_name
        create_dir_3(dir_server_dataset)
        for fine_dir_name in os.listdir(user_dir + "/" + x_folder_subdir[0]):
            create_dir_2(user_dir, fine_dir_name)
            create_dir_3(dir_server_dataset + "/" + fine_dir_name)
            fine_dir = user_dir + "/" + x_folder_subdir[0] + "/" + fine_dir_name + "/" + "saved_model"
            if os.path.isdir(fine_dir):
                saved_model_path = server_dir + "/"  + usr_dir_name + "/" + fine_dir_name + "/" + fine_dir_name + "/" + "1"
                if os.path.exists(saved_model_path):
                    try:
                        shutil.rmtree(saved_model_path, ignore_errors=True)
                    except OSError as e:
                        print("Error: %s : %s" % (saved_model_path, e.strerror))
                if not os.path.exists(saved_model_path):
                    src = fine_dir
                    dest = saved_model_path
                    shutil.copytree(src, dest) 

                temp_dict = {}
                name = user_dir.split('/')[-1] + "_" + fine_dir_name.split("-")[0] + "_" + fine_dir_name.split("_")[-1]
                temp_dict['name'] = name.lower()
                temp_dict['base_path'] = "/" + "models" + "/" + fine_dir_name 
                temp_dict['dataset'] = user_dir.split('/')[-1]
                temp_dict['model'] = fine_dir_name
                config_dict.append(temp_dict)

    # Writting into server config file
    config_dict_len = len(config_dict)
    loop = math.ceil(config_dict_len/2)
    diff = config_dict_len - loop 
    for i in range(config_dict_len):
        count_b += 1
        config_path = server_dir  + "/" + config_dict[i]['dataset'] + "/" + config_dict[i]['model'] + "/" + config_dict[i]['name'] + ".config"
        with open(config_path, "w") as config_file:
            s = "model_config_list {\n"
            s += "  config {\n"
            s += "    name: '{}'\n".format(config_dict[i]['name'])
            s += "    base_path: '{}'\n".format(config_dict[i]['base_path'])
            s += "    model_platform: 'tensorflow'\n"
            s += "  }\n"
            s += "}"
            config_file.write(s)

    model_dict = {
            'faster': 'Faster-RCNN-ResNet-50',
            'ssd': 'SSD-EfficientDet',
            'mask': 'Mask-RCNN'
        }

    # Creating csv file 
    count_b = 0
    csv_fpath = csv_dir + "/" + "TF_Server_Info.csv"
    # Create Panda Dataframe to record evaluated result
    server_df = pd.DataFrame(columns = ["User Dataset", "Model", "Index", "GRPC Port", "REST Port", "URL", "Config File Name"])
    for i in range(config_dict_len):
        port += 1
        count_b += 1
        config_path = server_dir + "/" + config_dict[i]['dataset'] + "/" + config_dict[i]['model'] + "/" + config_dict[i]['name'] + ".config"
        config = config_dict[i]
        user_dataset = config['name'].split('_')[0]
        model = config['model']
        index = config['name'].split('_')[2]
        url = "http://localhost:{}/v1/models/{}:predict".format(port, config['name'])
        new_row = {"User Dataset": user_dataset, "Model": model, "Index": index, "GRPC Port": port-100, "REST Port": port, "URL": url, "Config File Name": config_path}
        server_df = server_df.append(new_row, ignore_index=True)
        
        # Batch file path
        batch_dir = server_dir + "/" + config_dict[i]['dataset'] + "/" + config_dict[i]['model'] 
        batch_create_fpath = batch_dir + "/create_server.bat"
        batch_start_fpath = batch_dir + "/start_server.bat"
        batch_kill_fpath = batch_dir + "/kill_server.bat"
        saved_model_dir = batch_dir + "/" + config['model']

        # Write into batch file
        # 1. Create Docker Image batch file
        with open(batch_create_fpath, "w") as batch_create_file:
            # docker run -d -p 8601:8601 --name server_1 -t tensorflow/serving --rest_api_port=8601 --allow_version_labels_for_unavailable_models --model_config_file=/models/server_1.config
            s = "docker run -d -p {}:8500 -p {}:{} --name {}_ct -t tensorflow/serving --rest_api_port={} --allow_version_labels_for_unavailable_models --model_config_file=/models/{}.config\n".format(
                port-100, port, port, config['name'], port, config['name'])
            s += "docker cp {} {}:/models/{}\n".format(saved_model_dir, config['name']+"_ct", config['model'])
            s += "docker cp {} {}:/models/{}.config\n".format(config_path, config['name']+"_ct", config['name'])
            s += 'docker commit --change "ENV MODEL_NAME {}.config" {} {}\n'.format(config['name'], config['name']+"_ct", config['name'])
            s += 'docker container rm {}\n'.format(config['name']+"_ct")
            s += 'exit'
            batch_create_file.write(s)
        # 2. Start Container batch file
        with open(batch_start_fpath, "w") as batch_start_file:
            s = "start cmd /k docker run -p {}:8500 -p {}:{} --name {} {}".format(port-100, port, port, config['name'], config['name'])
            batch_start_file.write(s)
        # 3. Kill Container and Image batch file
        with open(batch_kill_fpath, "w") as batch_kill_file:
            s = "docker container stop {}\n".format(config['name'])
            s += "docker container rm -f {}\n".format(config['name'])
            s += "docker image rm {}\n".format(config['name'])
            s += 'exit'
            batch_kill_file.write(s)
    # Save Evaluation result CSV
    server_df.to_csv(csv_fpath, index=False, index_label=False)



if __name__ == '__main__':
    now = datetime.now()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Executing script_0.py\n")
    main()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Completed execution of script_0.py\n")












