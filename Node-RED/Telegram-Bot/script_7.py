import os
import glob as glob
import pandas as pd
import argparse
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
streamlit_user_dir = args.streamlit_user_dir_fpath
root_dir_list = args.streamlit_user_dir_fpath.split('/')
root_dir_list.pop()
root_dir = '/'.join(root_dir_list)

def main():
    
    csv_path = root_dir + "/user_csv/user_model_log.csv"
    user_model_log_df = pd.DataFrame(columns=['Dataset', 'Model', 'Average Accuracy',
                                              'Average Detection Time (s)', 'Average Response Time (s)',
                                              'Number of Usage'])
    for dataset_name in os.listdir(streamlit_user_dir):
        dataset_dir = streamlit_user_dir + "/" + dataset_name
        for model_name in os.listdir(dataset_dir + "/node-red"):
            csv_log_fpath = dataset_dir + "/node-red/" + model_name + "/csv_prediction_log/overall_result.csv"
            if os.path.exists(csv_log_fpath):
                csv_log_df = pd.read_csv(csv_log_fpath) #'Average Accuracy', 'Average Detection Time (s)', 'Average Response Time (s)'
                average_accuracy = csv_log_df.iloc[0]['Average Accuracy']
                average_detection_time = csv_log_df.iloc[0]['Average Detection Time (s)']
                average_response_time = csv_log_df.iloc[0]['Average Response Time (s)']
                number_of_usage = csv_log_df.iloc[0]['Number of Usage']
                temp_dict = {'Dataset': dataset_name, 'Model': model_name, 'Average Accuracy': average_accuracy,
                             'Average Detection Time (s)': average_detection_time, 'Average Response Time (s)': average_response_time,
                             'Number of Usage': number_of_usage}
                user_model_log_df = user_model_log_df.append(temp_dict, ignore_index=True)
    user_model_log_df.to_csv(csv_path, index=False, index_label=False)

if __name__=="__main__":
    now = datetime.now()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Executing script_7.py\n")
    main()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Completed execution of script_7.py\n")