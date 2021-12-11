import os
import pandas as pd
import argparse
from datetime import datetime


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Check model path.", allow_abbrev=False)
parser.add_argument("-e",
                    "--eval_log_file",
                    help="Path to the specified eval_log_file in the csv_prediction_log folder in specified dataset folder.",
                    type=str,
                    default=None, required=True)
parser.add_argument("-r",
                    "--row_number",
                    help="Row number of edited cell in Node-RED.",
                    type=int,
                    default=None, required=True)
parser.add_argument("-c",
                    "--user_count",
                    help="New value of User_Count updated in Node-RED",
                    type=int,
                    default=None, required=True)

args = parser.parse_args()

def main():
    
    user_count = args.user_count
    row_number = args.row_number
    csv_path = args.eval_log_file

    split_list = args.eval_log_file.split('/')
    split_list.pop()
    join_list = '/'.join(split_list)

    overall_result_csv_path = join_list + "/overall_result.csv"
    csv_df = pd.read_csv(csv_path)

    pred_count = csv_df.iloc[row_number]['Count']
    csv_df.at[row_number, 'User_Count'] = user_count

    if (pred_count>user_count):
        csv_df.at[row_number, 'Accuracy'] = round((pred_count - (abs(pred_count-user_count)))/pred_count, 4)
        csv_df.at[row_number, 'User Remark'] = "no"
    elif (pred_count<user_count):
        csv_df.at[row_number, 'Accuracy'] = round(pred_count/user_count, 4)
        csv_df.at[row_number, 'User Remark'] = "no"
    elif(pred_count==user_count):
        csv_df.at[row_number, 'User Remark'] = "yes"
        csv_df.at[row_number, 'Accuracy'] = 1.0000

    csv_df.to_csv(csv_path, index=False, index_label=False)

    # Writting into overall result CSV file
    overall_result_pd = pd.DataFrame(columns = ['Average Accuracy', 'Average Detection Time (s)', 'Average Response Time (s)', 'Number of Usage'])
    csv_df = pd.read_csv(csv_path)
    total_accuracy, total_num_row = csv_df['Accuracy'].sum(), len(csv_df.index)
    average_accuracy = round((total_accuracy/total_num_row), 4)
    # Evaluate average detection time per object in an image
    average_detection_time = round((csv_df['Time Per Detection (s)'].sum())/total_num_row, 4)
    # Evaluate average TensorFlow Server Response time
    average_response_time = round((csv_df['Response Time (s)'].sum())/total_num_row, 4)
    number_of_usage = len(csv_df['Image Name'].unique())
    # Writting into CSV
    temp_dict = {'Average Accuracy': average_accuracy, 'Average Detection Time (s)': average_detection_time, 'Average Response Time (s)': average_response_time,
                 'Number of Usage': number_of_usage}
    overall_result_pd = overall_result_pd.append(temp_dict, ignore_index=True)
    overall_result_pd.to_csv(overall_result_csv_path, index=False, index_label=False)



if __name__=="__main__":
    now = datetime.now()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Executing script_8.py\n")
    main()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Completed execution of script_8.py\n")
