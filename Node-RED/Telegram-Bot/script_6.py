import os
import pandas as pd
import argparse
from datetime import datetime


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Check model path.", allow_abbrev=False)
parser.add_argument("-n",
                    "--node_red_directory",
                    help="Path to the node-red directory in streamlit user directory.",
                    type=str,
                    default=None, required=True)
parser.add_argument("-i",
                    "--img_file_name",
                    help="Name of image.",
                    type=str,
                    default=None, required=True)
parser.add_argument("-r",
                    "--user_remark",
                    help="User remark of image count.",
                    type=str,
                    default=None, required=True)

args = parser.parse_args()

def main():
    csv_path = args.node_red_directory + "csv_prediction_log" + "/eval_log_file.csv"
    overall_result_csv_path = args.node_red_directory + "csv_prediction_log" + "/overall_result.csv"
    csv_df = pd.read_csv(csv_path)
    img_filename = args.img_file_name
    remark = args.user_remark
    grouped = csv_df.groupby('Image Name')
    for key, group_df in grouped:
        if key==img_filename:
            temp_df = group_df
            temp_df = temp_df.fillna(value={'User Remark': remark})
            csv_df.drop(csv_df.index[csv_df['Image Name']==key], inplace=True)
            for i, row in group_df.iterrows():
                if remark=='yes':
                    temp_df.at[i, 'User_Count'] = row['Count']
                    temp_df.at[i, 'Accuracy'] = 1.0
                    temp_df.at[i, 'User Remark'] = 'yes'
                elif remark=='no':
                    temp_df.at[i, 'User_Count'] = 0.0
                    temp_df.at[i, 'Accuracy'] = 0.0
                    temp_df.at[i, 'User Remark'] = 'no'
            csv_df = csv_df.append(temp_df, ignore_index=True)
            break
    csv_df.to_csv(csv_path, index=False, index_label=False)

    # Writting into overall result CSV file
    overall_result_pd = pd.DataFrame(columns = ['Average Accuracy', 'Average Detection Time (s)', 'Average Response Time (s)', 'Number of Usage'])
    csv_df = pd.read_csv(csv_path)
    total_accuracy, total_num_row = csv_df['Accuracy'].sum(), len(csv_df.index)
    num_usage = len(csv_df['Image Name'].unique())
    average_accuracy = round(total_accuracy/total_num_row, 2)
    # Evaluate average detection time per object in an image
    average_detection_time = round((csv_df['Time Per Detection (s)'].sum())/total_num_row, 4)
    # Evaluate average TensorFlow Server Response time
    average_response_time = round((csv_df['Response Time (s)'].sum())/total_num_row, 4)
    # Writting into CSV
    temp_dict = {'Average Accuracy': average_accuracy, 'Average Detection Time (s)': average_detection_time, 'Average Response Time (s)': average_response_time,
                 'Number of Usage': num_usage}
    overall_result_pd = overall_result_pd.append(temp_dict, ignore_index=True)
    overall_result_pd.to_csv(overall_result_csv_path, index=False, index_label=False)



if __name__=="__main__":
    now = datetime.now()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Executing script_6.py\n")
    main()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Completed execution of script_6.py\n")
