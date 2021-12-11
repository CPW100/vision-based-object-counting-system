import os
import glob
import json
import argparse
import regex as re
import pandas as pd


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Check model path.", allow_abbrev=False)
parser.add_argument("-u",
                    "--streamlit_user_dir_fpath",
                    help="Path to the folder where Streamlit user directory is stored.",
                    type=str,
                    default=None, required=True)

args = parser.parse_args()
x_folder_subdir = ("fine_tuned", "node-red")
streamlit_user_dir = args.streamlit_user_dir_fpath
dir_list = streamlit_user_dir.split("/")
dir_list.pop()
rejoin_path = "/".join(dir_list)

def main():
    csv_dir = rejoin_path + "/" + "user_csv"
    server_df = pd.read_csv(csv_dir + "/TF_Server_Info.csv")
    output_list = []
    grouped = server_df.groupby('User Dataset')
    for key, group_df in grouped:
        output_dict = {}
        output_dict[key] = []

        for i, row in group_df.iterrows():
            name = row['User Dataset'] + "_" + row['Model'].split("-")[0] + "_" + str(row["Index"]) 
            server_name = name.lower()
            model_file_name = row['Model'] 
            label_map_fpath = rejoin_path + "/streamlit_user_directory" + "/" + key + "/" + "dataset" + "/" + key + "_label_map.pbtxt"
            node_red_dir = rejoin_path + "/streamlit_user_directory" + "/" + key + "/" + "node-red" + "/" + model_file_name + "/"
            temp_dict = {}
            temp_dict[row['Model']] = {}
            temp_dict[row['Model']]['grpc_port'] = row['GRPC Port']
            temp_dict[row['Model']]['server_name'] = server_name
            temp_dict[row['Model']]['server_url'] = row['URL']
            temp_dict[row['Model']]['node_red_directory'] = node_red_dir
            temp_dict[row['Model']]['label_map_file_path'] = label_map_fpath
            output_dict[key].append(temp_dict)
        output_list.append(output_dict)

    output_json = json.dumps(output_list)
    print(output_json)

if __name__ == '__main__':
    rejoin_path = args.streamlit_user_dir_fpath.split('/')
    rejoin_path.pop()
    rejoin_path = '/'.join(rejoin_path) 
    main()



