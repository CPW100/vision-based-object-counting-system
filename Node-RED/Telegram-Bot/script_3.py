import os
import pathlib
import requests
import glob
import random
import time
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import io
import json
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import cv2
import json
import streamlit as st
import argparse
import paho.mqtt.client as mqtt


# Custom import
import visualization_utils as viz_utils

from fractions import Fraction
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils_test as viz_utils_test
#from object_detection.builders import model_builder
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from datetime import datetime
from collections import Counter
from grpc_request import grpc_request

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Evaluate and Count Deep Learning script.", allow_abbrev=False)
parser.add_argument("-u",
                    "--server_url",
                    help="URL used to deploy custom trained modelusing Tensorflow Serving API.",
                    type=str,
                    default=None, required=True)
parser.add_argument("-n",
                    "--node_red_directory",
                    help="Path to the model checkpoints.",
                    type=str,
                    default=None, required=True)
parser.add_argument("-l",
                    "--label_map_file_path",
                    help="Path to the custom dataset label map.",
                    type=str,
                    default=None, required=True)
parser.add_argument("-m",
                    "--model_name",
                    help="Model name in TensorFlow Server.",
                    type=str,
                    default=None, required=True)
parser.add_argument("-g",
                    "--grpc_port",
                    help="GRPC Port in Tensorflow Server",
                    type=str,
                    default=None, required=True)
parser.add_argument('-i', '--image_name_list', nargs='+', default=[], required=True)
args = parser.parse_args()


class Evaluate_and_Count:
    def __init__(self, server_url, label_map_file_path, node_red_directory, model_name, grpc_port, image_name_list):

        self.grpc_port = grpc_port
        self.model_name = model_name
        self.label_map_file_path = label_map_file_path
        self.server_url = server_url
        self.evaluation_img_fpath = node_red_directory + 'bbox_img'
        self.evaluation_csv_path = node_red_directory + 'csv_prediction_log'
        self.test_img_fpath = node_red_directory + 'telegram_img'
        self.image_name_list = image_name_list
        self.test_img_path_list = []
        self.result = []
        for v in range(len(self.image_name_list)):
            test_img_fpath = self.test_img_fpath + "/" + self.image_name_list[v] + '.jpg'
            self.test_img_path_list.append(test_img_fpath)
        

        # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime('%d/%m/%Y %H:%M:%S')

        print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Connected with result code "+str(rc) + "\n")

        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("test")
        client.subscribe("single_img_data")
        client.subscribe("all_img_data")
        client.subscribe("eval_status")

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        print("   " + msg.topic+" "+str(msg.payload))

    # The callback when the client disconnect from the broker
    def on_disconnect(self, client, userdata, rc):
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime('%d/%m/%Y %H:%M:%S')

        print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Disconnected from broker.\n")

    # The callback when the client publish a message
    def on_publish(self, client, userdata, mid):
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime('%d/%m/%Y %H:%M:%S')

        print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Message is published.\n")
        print(f"    client: {client}\n")
        print(f"    userdata: {userdata}\n")
        print(f"    mid: {mid}\n")

    # Define function
    def get_model_detection_function(self, model):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)
    
            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    def load_image_into_numpy_array(self, img_path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
          path: the file path to the image

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(Image.open(img_path))

    def create_eval_log_file(self):
        # Check if the csv file exists
        evaluation_result_csv_fpath = self.evaluation_csv_path + "/" + "eval_log_file.csv"
        if os.path.exists(evaluation_result_csv_fpath):
            pass
        else:
            # Create Panda Dataframe to record evaluated result
            evaluation_result_pd = pd.DataFrame(columns = ['Date', 'Time',
                                                          'Image Name', 'Class', 'Count',
                                                          'Time Per Detection (s)', 'Response Time (s)', 'User Remark', 
                                                          'User Count', 'Accuracy'])
            # Save Evaluation result CSV
            evaluation_result_pd.to_csv(evaluation_result_csv_fpath, index=False, index_label=False)
        return evaluation_result_csv_fpath

    def remove_duplicates_in_log_file(self):
        evaluation_result_csv_fpath = self.create_eval_log_file()
        evaluation_result_pd = pd.read_csv(evaluation_result_csv_fpath)
        temp_df = evaluation_result_pd.sort_values('Time').drop_duplicates('Image Name',keep='last')
        temp_df.to_csv(evaluation_result_csv_fpath, index=False, index_label=False)

    def call_MQTT(self):
        # 'Image Name', 'Class', 'Count', 'Time Per Detection (s)'

        # Read CSV file to get count result
        evaluation_result_csv_fpath = self.create_eval_log_file()
        evaluation_result_pd = pd.read_csv(evaluation_result_csv_fpath)
        grouped = evaluation_result_pd.groupby(['Image Name'])

        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime('%d/%m/%Y %H:%M:%S')

        # Initialize mqtt stuff
        broker_ip = "localhost" # local host
        broker_port = 1883
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_disconnect = self.on_disconnect
        client.on_message = self.on_message
        client.on_publish = self.on_publish

        # Connect to MQTT
        print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Connecting to MQTT broker.\n")
        client.connect(broker_ip, broker_port, 60)
        print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Connected to MQTT broker.\n")
        
        #client.publish(topic='test', payload="Hello, I am a Python client publisher.")

        # Read and convert image into byte array
        data_list = []
        for key, item in grouped:
            for i, img_path in enumerate(self.image_name_list):
                if key==img_path:
                    data_dict = {} 
                    data_dict[key] = {}
                    data_dict[key]['class'] = {}
                    sub_group = grouped.get_group(key)
                    unq_className_arr = sub_group['Class'].unique()
                    for unq_name in unq_className_arr:
                        extracted_row = sub_group.loc[sub_group['Class']==unq_name]
                        print(extracted_row)
                        print(data_dict[key]['class'])
                        data_dict[key]['class'][unq_name] = extracted_row.Count.item()

                    data_dict[key]['abs_path'] = self.evaluation_img_fpath + "/" + self.image_name_list[i] + '_bb.jpg'
                    data_list.append(data_dict)

                    data_dict_json = json.dumps(data_dict)
        #            #my_img = Image.open(img_path)
        #            #buf = io.BytesIO()
        #            #my_img.save(buf, format='JPEG')
        #            #byte_im = buf.getvalue()
                    client.loop_start()
                    client.publish(topic='single_img_data', payload=data_dict_json)
                    client.publish(topic='eval_status', payload="done")
                    client.loop_stop()
        #            #client.publish(topic='all_img_data', payload=byte_im)
                    time.sleep(4)
        data_list_json = json.dumps(data_list)
        client.loop_start()
        client.publish(topic='all_img_data', payload=data_list_json)
        client.loop_stop()
        # Disconnect from MQTT
        client.disconnect()


    def start(self):
        now = datetime.now()
        # Image list to be evaluated
        img_list = self.image_name_list

        #map labels for inference decoding
        label_map = label_map_util.load_labelmap(self.label_map_file_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

        #run detector on test image
        #it takes a little longer on the first run and then runs at normal speed. 
        img_type = 'jpg'
        TEST_IMAGE_PATHS = self.test_img_path_list
        batch = len(TEST_IMAGE_PATHS)

        
        

        for i in range(batch):

            total_detection = 0
            time_per_detection = 0
            time_taken = 0
            evaluation_result_csv_fpath = self.create_eval_log_file()
            evaluation_result_pd = pd.read_csv(evaluation_result_csv_fpath)

            new_row = []
            kv_str = {}
            test_label_list = []
            image_path = TEST_IMAGE_PATHS[i]
            image_np = self.load_image_into_numpy_array(image_path)

            # GRPC request
            GRPC_Request = grpc_request(image_np, self.grpc_port, self.model_name)
            response_time, detection_boxes, detection_classes, detection_scores = GRPC_Request.main()
            print(f'took {response_time}s')
            print(detection_classes)

            # RESTful request
            #payload = {"instances": [image_np.tolist()]}
            #num_requests = 20
            #start_time = time.time()
            #for i in range(num_requests):
            #    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: loop number --> {i}\n")
            #    response = requests.post(self.server_url, json=payload)
            #    output_dict = response.json()
            #    key_list = list(output_dict.keys())
            #    print(key_list)
            #    if key_list[0]=="error":
            #        pass
            #    else:
            #        break
            #end_time = time.time()
            #print(f'took {end_time - start_time}s')
            #print(response.json())
            #predictions = response.json()['predictions'][0]
            
            ##key = list(predictions.keys())
            ##print(list(predictions.keys()))

            #detection_boxes = predictions['detection_boxes']
            #detection_classes = predictions['detection_classes']
            #detection_scores = predictions['detection_scores']            

            image_np_with_detections = image_np.copy()

            #img, my_label_list = viz_utils.visualize_boxes_and_labels_on_image_array(
            #    image_np_with_detections,
            #    np.array(detection_boxes),
            #    (np.array(detection_classes)).astype(int),
            #    np.array(detection_scores),
            #    category_index,
            #    # instance_masks=detections.get('detection_masks'),
            #    fontsize=13,
            #    use_normalized_coordinates=True,
            #    max_boxes_to_draw=20,
            #    min_score_thresh=.75,
            #    agnostic_mode=False,
            #    line_thickness=3)

            model_type = self.model_name.split("_")[1]
            if model_type=="mask":
                n=100
            elif model_type=='faster':
                n=300
            else:
                n=100

            img, my_label_list = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                np.reshape(detection_boxes,[n,4]),
                np.squeeze(detection_classes).astype(np.int32),
                np.squeeze(detection_scores),
                category_index,
                fontsize=7,
                use_normalized_coordinates=True,
                max_boxes_to_draw=100,
                min_score_thresh=.85,
                agnostic_mode=False,
                line_thickness=3)
            print(my_label_list)

            for element in my_label_list:
                label = element[0][0].split(':')[0]
                test_label_list.append(label)

            test_count = Counter(test_label_list)
            for key,value in test_count.items():
                kv_str[key] = value
                total_detection += value
        
            # Saving image
            img_fname = image_path.split("/")[-1].split(".")[0]
            im = Image.fromarray(image_np_with_detections)
            save_img_path = self.evaluation_img_fpath + "/" + img_fname + "_bb" + ".jpg"
            im.save(save_img_path)

            # Calculate time taken per image
            #time_taken = round(end_time - start_time,2)
            time_per_detection = round(response_time/total_detection, 2)

            # Append data into the csv file
            # 'Image Name', 'Class', 'Count', 'Total Detection Count', 'Time Per Detection'
            # datetime object containing current date and time
            now = datetime.now()
            today_date = now.strftime('%d/%m/%Y')
            current_time = now.strftime('%H:%M:%S')
            for class_key, class_count in kv_str.items():
                temp_dict = {'Date': None, 'Time': None, 'Image Name': None, 'Class': None, 
                             'Count': None, 'Time Per Detection (s)': None, 'Response Time (s)': None, 'User Remark': None, 
                             'User Count': None, 'Accuracy': None}
                temp_dict['Date'] = today_date
                temp_dict['Time'] = current_time
                temp_dict['Image Name'] = img_fname
                temp_dict['Class'] = class_key
                temp_dict['Count'] = class_count
                temp_dict['Time Per Detection (s)'] = time_per_detection
                temp_dict['Response Time (s)'] = response_time
                new_row.append(temp_dict)

            # Clear dictionary
            kv_str = ""
            # Save Evaluation result CSV
            for j in range(len(new_row)):
                evaluation_result_pd = evaluation_result_pd.append(new_row[j], ignore_index=True)
            evaluation_result_pd.to_csv(evaluation_result_csv_fpath, index=False, index_label=False)
            # self.remove_duplicates_in_log_file()

        
if __name__ == '__main__':
    now = datetime.now()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Started evaluation process.\n")
    eval_and_count = Evaluate_and_Count(args.server_url, args.label_map_file_path, args.node_red_directory, args.model_name, args.grpc_port, args.image_name_list)
    eval_and_count.start()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: MQTT in action.\n")
    eval_and_count.call_MQTT()
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Completed evaluation process.\n")








