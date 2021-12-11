import os
import pathlib
import glob
import random
import time
import sexporter_main_v2 as exporter_main_v2
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import cv2
import streamlit as st

# Custom import
import svisualization_utils as viz_utils

from fractions import Fraction
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils_test as viz_utils_test
from object_detection.builders import model_builder
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from collections import Counter


class Evaluate_and_Count:
    def __init__(self, test_dataset_csv_fpath, custom_pipeline_config_path, training_dir, evaluation_img_fpath, evaluation_csv_path, test_img, test_img_name):
        self.test_dataset_csv_fpath = test_dataset_csv_fpath
        self.custom_pipeline_config_path = custom_pipeline_config_path
        self.training_dir = training_dir
        self.evaluation_img_fpath = evaluation_img_fpath
        self.evaluation_csv_path = evaluation_csv_path
        self.test_img = test_img
        self.test_img_name = test_img_name

    def initiate_session_state(self):
        # Initializing Session State
        session_state_keys = {"prog_bar_status": None}
        for key, value in session_state_keys.items():
            if key not in st.session_state:
                st.session_state[key] = value

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

    def load_image_into_numpy_array(self, img):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
          path: the file path to the image

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(img)

    def start(self):

        # Initiate progress bar
        self.initiate_session_state()
        prog_placeholder = st.empty()
        st.session_state["prog_bar_status"] = "Starting to evaluate..."
        prog_placeholder.text(st.session_state["prog_bar_status"])
        my_bar = st.progress(0)

        # Create Panda Dataframe to record evaluated result
        evaluation_result_pd = pd.DataFrame(columns = ['Image Name', 'Prediction', 'False Positive', 'Class', 'Prediction Count', 'Actual Count'])
        average_eval_result_pd = pd.DataFrame(columns = ['Accuracy', 'False Positive Detection', 'Total time taken (s)', 'Time per detection (s)'])

        filenames = list(pathlib.Path(self.training_dir).glob('*.index'))

        #recover our saved model
        pipeline_config = self.custom_pipeline_config_path
        #generally you want to put the last ckpt from training in here
        model_dir = str(filenames[-1]).replace('.index','')    # /gdrive/Shareddrives/DeepLearn/1-FCNN/training_1/ckpt-15
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(str(filenames[-1]).replace('.index','')))

        # Load model
        detect_fn = self.get_model_detection_function(detection_model)

        #map labels for inference decoding
        label_map_path = configs['eval_input_config'].label_map_path
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

        #run detector on test image
        #it takes a little longer on the first run and then runs at normal speed. 
        total_detection, total_gt = 0,0
        img_type = 'jpg'
        correct_pred_acc, wrong_pred_acc = [], []
        avg_accuracy = []
        # Read CSV file to obtain number of objects in the image
        test_csv = pd.read_csv(self.test_dataset_csv_fpath)
        # TEST_IMAGE_PATHS = glob.glob(os.path.join(self.test_dataset_img_fpath,'*.' + img_type))
        batch = len(self.test_img)
        random_int_list = random.sample(range(batch), batch)
        plt.figure(figsize=(50,120)) # figsize=(40,20)

        accuracy = 0
        false_positive = 0
        time_per_detection = 0
        time_taken = 0

        for k in range(batch):
            kv_str = {}
            test_label_list = []
            image_path = self.test_img[k]
            image_np = self.load_image_into_numpy_array(self.test_img[k])

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            start_time = time.time()
            detections, predictions_dict, shapes = detect_fn(input_tensor)
            end_time = time.time()

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            img, my_label_list = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                # instance_masks=detections.get('detection_masks'),
                fontsize=10,
                use_normalized_coordinates=True,
                max_boxes_to_draw=100,
                min_score_thresh=.98,
                agnostic_mode=False,
                line_thickness=3)

            for element in my_label_list:
                label = element[0][0].split(':')[0]
                test_label_list.append(label)

            test_count = Counter(test_label_list)
            unique_class_list = list(test_count.keys())
            predict_class_dict = {}
            compare_class_dict = {}
            for test_class in unique_class_list:
                predict_class_dict[test_class] = 0;
                compare_class_dict[test_class] = 0;
            for key,value in test_count.items():
                kv_str[key] = value
                total_detection += value
            img_fname = self.test_img_name[k]
            img_csv = test_csv.loc[test_csv['filename'] == img_fname]
            img_centre_csv = img_csv.drop(['width','height'], axis=1)
            img_centre_csv = img_centre_csv.assign(x_centre=np.zeros((1,len(img_centre_csv)),dtype=int)[0].tolist(),
                                                y_centre=np.zeros((1,len(img_centre_csv)),dtype=int)[0].tolist(),
                                                bbox_area=np.zeros((1,len(img_centre_csv)),dtype=int)[0].tolist())
        
            iter = 0
            for idx, row in img_centre_csv.iterrows():
                xmin = row['xmin']
                xmax = row['xmax']
                ymin = row['ymin']
                ymax = row['ymax']
    
                # compute centre of bounding box
                x_centre = int((xmin + xmax)/2)
                y_centre = int((ymin + ymax)/2)
                box_area = (abs(xmin-xmax))*(abs(ymin-ymax))
                img_centre_csv.iloc[iter,6] = x_centre
                img_centre_csv.iloc[iter,7] = y_centre
                img_centre_csv.iloc[iter,8] = box_area
                iter += 1
            
            # Calculate accuracy
            gt_total = len(img_centre_csv)
            md_total = len(my_label_list)
            correct_prediction, wrong_prediction = 0,0
            compare_log = np.zeros(gt_total, dtype='int')
            predict = 0
            for i, my_test_list in enumerate(my_label_list):
                my_test_label =  my_test_list[0][0].split(':')[0]
                my_test_x_centre, my_test_y_centre = int(my_test_list[1][0]), int(my_test_list[1][1])
                my_test_bbox_area = int(my_test_list[2])
                test_xmin, test_xmax, test_ymin, test_ymax = int(my_test_list[3][0]), int(my_test_list[3][1]), int(my_test_list[3][2]), int(my_test_list[3][3])
                count = 0
                for idx, row in img_centre_csv.iterrows():
                    org_x_center, org_y_center, org_area, org_xmin, org_xmax, org_ymin, org_ymax = row['x_centre'], row['y_centre'], row['bbox_area'], row['xmin'], row['xmax'], row['ymin'], row['ymax']
                    diff_x_center, diff_y_center, diff_area_probability = abs(org_x_center-my_test_x_centre), abs(org_y_center-my_test_y_centre), abs(my_test_bbox_area-org_area)/org_area
                    area_probability = [abs(1-diff_area_probability) if (diff_area_probability>1) else diff_area_probability]
                    diff_xmin, diff_xmax, diff_ymin, diff_ymax = abs(org_xmin-test_xmin), abs(org_xmax-test_xmax), abs(org_ymin-test_ymin), abs(org_ymax-test_ymax)
                    org_label = row['class']

                    if (org_label==my_test_label and diff_x_center<=15 and diff_y_center<=15 and diff_xmin<=15 and diff_xmax<=15 and diff_ymin<=15 and diff_ymax<=15):
                      # st.write(f"my label list, img_csv_row: {i},{idx}")
                      # st.write(f"{org_xmin}:{test_xmin}, {org_xmax}:{test_xmax}, {org_ymin}:{test_ymin}, {org_ymax}:{test_ymax}")
                      # st.write(f"diff: {diff_xmin}, {diff_xmax}, {diff_ymin}, {diff_ymax}")
                      # st.write(f"first: {my_test_x_centre}, {my_test_y_centre}, {my_test_bbox_area}")
                      # st.write(f"second: {org_x_center}, {org_y_center}, {org_area}")
                      # st.write(f"diff x: {diff_x_center}, diff y: {diff_y_center}, diff area: {diff_area_probability}\n")
                        
                        for compare_key, compare_value in compare_class_dict.items():
                            if compare_key==org_label:
                                compare_class_dict[compare_key] = compare_value + 1
                        compare_log[count] +=1
                        if (compare_log[count]==1):
                            correct_prediction += 1
                            for predict_key, predict_value in predict_class_dict.items():
                                if predict_key==org_label:
                                    predict_class_dict[predict_key] = predict_value + 1 
                    count += 1       
            # st.write(compare_log)
            # st.write(my_label_list)
            # st.write(img_centre_csv)
            wrong_prediction = abs(md_total-compare_log.sum())
            
            total_gt += gt_total
            correct_pred_acc.append(correct_prediction)
            wrong_pred_acc.append(wrong_prediction)

            final_prediction = round(correct_prediction/gt_total, 3)
            final_false_positive = round(wrong_prediction/len(my_label_list), 3)
            img_centre_df = img_centre_csv.groupby(['class']).count()
            for obj_class, obj_count in kv_str.items():
                for pred_key, pred_count in predict_class_dict.items():
                        for comp_key, comp_count in compare_class_dict.items():
                            if obj_class==pred_key==comp_key:
                                gt_count = img_centre_df['filename'][pred_key]
                                final_pred = round(pred_count/gt_count, 4)
                                final_fp = round((abs(obj_count-comp_count))/obj_count,4)
                                evaluation_result_pd = evaluation_result_pd.append({'Image Name' : img_fname, 'Prediction' : final_pred, 
                                                                                    'False Positive' : final_fp, 'Class': obj_class, 
                                                                                    'Prediction Count': obj_count, 'Actual Count': gt_count}, 
                                                                                    ignore_index = True)

            # Saving image
            im = Image.fromarray(image_np_with_detections)
            save_img_path = self.evaluation_img_fpath + "/" + img_fname.split('.')[0] + "_bounded" + ".jpg"
            im.save(save_img_path)
            kv_str = ""

            # Update progress bar
            prog_placeholder.empty()
            st.session_state["prog_bar_status"] = "Evaluating {} of {} | Completion Percentage: {}%".format(k+1, batch, int(((k+1)/batch)*100))
            prog_placeholder.text(st.session_state["prog_bar_status"])
            my_bar.progress(int(((k+1)/batch)*100))
            
            # Calculate time taken  
            time_Taken = round(end_time - start_time,4)
            time_taken += time_Taken
            
        time_per_detection = round(time_taken/total_detection, 4)
        avg_total_time = round(time_taken/batch, 4)
        # Calculate the average accuracy
        accuracy = sum(correct_pred_acc)/total_gt
        false_positive = sum(wrong_pred_acc)/total_detection
        final_test_acc = round(accuracy, 4)
        final_false_positive_rate = round(false_positive,4)
        average_eval_result_pd = average_eval_result_pd.append({'Accuracy' : final_test_acc, 'False Positive Detection' : final_false_positive_rate, 
                                                            'Total time taken (s)' : avg_total_time, 'Time per detection (s)': time_per_detection},
                                                                ignore_index = True)

        # Save Evaluation result CSV
        evaluation_result_csv_fpath = self.evaluation_csv_path + "/" + "individual_img_eval_result.csv"
        average_eval_result_csv_fpath = self.evaluation_csv_path + "/" + "average_eval_result.csv"
        evaluation_result_pd.to_csv(evaluation_result_csv_fpath, index=False, index_label=False)
        average_eval_result_pd.to_csv(average_eval_result_csv_fpath, index=False, index_label=False)






