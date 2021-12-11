import streamlit as st
import os
import sys
import io
import json
import xmltodict
import pandas as pd
import requests
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

from PIL import Image
from stfrecord import *
from zipfile import ZipFile
from tqdm import tqdm
#from stqdm import stqdm
from collections import namedtuple
from object_detection.utils import dataset_util, label_map_util

def app():
    # Making directories
    main_dir = os.getcwd()
    split_path = main_dir.split("/")
    split_path = split_path[:len(split_path)-1]
    separator = "/"
    rejoin_path = separator.join(split_path)
    streamlit_user_directory = rejoin_path + "/" + "streamlit_user_directory"
    x_folder_subdir = ("dataset", "training", "fine_tuned", "pipeline", "evaluation", "node-red")


    # Streamlit Interface
    st.subheader("""Upload Dataset""")

    # Initializing Session State
    session_state_keys = {'streamlit_user_dir_list': None, 'user_dir_choice': None, 'empty_container_1': None, 
                          'xml_train': None,  'xml_test': None, 'number_of_label': None, 'labels': None, 'uploaded_img_train': None, 
                          'uploaded_img_test': None, 'img_encoded_train': None, 'imgs_name_train': None, 'image_train': None, 
                          'img_encoded_test': None, 'imgs_name_test': None, 'image_test': None, 'TF_Record_User_Selection': None,
                          'xml_df_train': None, 'xml_df_test': None}
    for key, value in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value

    def clear_session_state():
        for key, value in session_state_keys.items():
            st.session_state[key] = value

    # 第一步驟：Generate directory
    def list_out_existing_user_directory(streamlit_user_directory):
        list_dir = []
        for filename in os.listdir(streamlit_user_directory):
            list_dir.append(filename)
        return list_dir
    def create_directory(path):   
        if not os.path.exists(path):
            os.mkdir(path)
    def generate_training_dir(x_folder_name, x_folder_subdir):
        x_path = rejoin_path + '/streamlit_user_directory/' + x_folder_name
        if not os.path.exists(x_path):
            print("Generating training directory")
            create_directory(x_path)
            for item in x_folder_subdir:
                create_directory(x_path + '/' + item)
            print(f"Training directory is at: {x_path}")

    def step_1_interface():
        st.info('**Step 1:** Generate user directory to save tf.record file, csv file and trained model.')
        # Check if streamlit_user_directory exist
        streamlit_user_directory = rejoin_path + '/streamlit_user_directory/'
        if not os.path.exists(streamlit_user_directory):
            os.mkdir(streamlit_user_directory)
        st.session_state["streamlit_user_dir_list"] = list_out_existing_user_directory(streamlit_user_directory)
        st.session_state["streamlit_user_dir_list"].append("+ Create a new directory")
        st.session_state["user_dir_choice"] =  st.selectbox(label="Please select or create a directory.",
                                                options=st.session_state["streamlit_user_dir_list"])
        st.session_state["empty_container_1"] = st.empty()
        # Define function for user's choice of "yes" or "no"
        if st.session_state["user_dir_choice"]=="+ Create a new directory":
            st.session_state["user_create_dir_name"]  = st.session_state["empty_container_1"].text_input(label="Enter a name for the folder to be created.")

        else: 
            st.session_state["empty_container_1"] = st.empty()

    # 第二步驟： Upload XML files from training datase. https://www.tensorflow.org/install/source#gpu
    def step_2_function(option=None):
        
        if option == "Train":
            key_1, key_2 = 'xml_train', "train"
        elif option == "Test":
            key_1, key_2 = 'xml_test', "test"

        st.session_state[key_1] = st.file_uploader(label="Upload XML files from {} dataset.".format(key_2), 
                                                            type=['xml'], accept_multiple_files=True)
        
    def step_2_interface():
        st.info('**Step 2:** Upload the XML files from train and test datasets.')
        col_upload_xml_train, col_upload_xml_test = st.columns(2)
        with col_upload_xml_train:
            step_2_function(option = "Train")

        with col_upload_xml_test:
            step_2_function(option = "Test")

    # Step 3. Generate label map
    def generate_label_map(label_map_output_path):
            if not os.path.exists(label_map_output_path):
                label_map_list = []
                labels = st.session_state['labels'].split(",")
                if st.session_state['labels']==None or st.session_state['labels']=="":
                    sys.exit("Please enter image label in step 2.")
                else:
                    if st.session_state['number_of_label'] == len(labels)-1:
                        st.session_state['error'] = None
                        for i in range(st.session_state['number_of_label']):
                            label_map_dict = {}
                            label_map_dict['name'] = labels[i]
                            label_map_dict['id'] = i+1
                            label_map_list.append(label_map_dict)

                        # Generate label_map.pbtxt
                        print(f"label map output path: {label_map_output_path}")
                        with open(label_map_output_path, 'w') as f:
                            for label in label_map_list:
                                f.write('item { \n')
                                f.write('\tname:\'{}\'\n'.format(label['name']))
                                f.write('\tid:{}\n'.format(label['id']))
                                f.write('}\n')
                        print("Label Map Status: Label map has been created successfully.")
                    else:
                        sys.exit("Please ensure the number of labels and the number of user input image labels are equivalent.")

    def step_3_interface():
        st.info('**Step 3:** Generate label map.')
        st.session_state['number_of_label'] = st.slider(label="Select number of labels", min_value=1, max_value=20, key="form_1_slider")
        st.session_state["labels"] = st.text_input(label="Enter image labels")
        st.markdown("""
        <section class="label_map_section" style="background-color:powderblue; padding: 10px 10px 10px 10px; border-radius: 5px 5px 5px 5px">
        <h3 style="text-align:left"><b>Example label input:</b></h3>
        <p>
        If there is only 1 label, enter: <i><b>carrot,</b></i><br>
        If there are multiple labels, enter: <i><b>carrot,rabbit,peony,rocket,</b></i><br>
        <strong>Remember to end the list with a comma and do not leave any space in between a word and comma!</strong>
        </p>
        </section>
        <p></p>
        """, unsafe_allow_html=True)

    # Step 4. Upload image
    def process_img(uploaded_image_files, option=None):
        if option=='train':
            key_1, key_2, key_3 = 'img_encoded_train', 'imgs_name_train', 'image_train'
        elif option=='test':
            key_1, key_2, key_3 = 'img_encoded_test', 'imgs_name_test', 'image_test'

        imgs, imgs_name, img_encoded = [], [], []
        if uploaded_image_files is not None:
            for img_file in uploaded_image_files:
                encoded_jpg = img_file.read()
                encoded_jpg_io = io.BytesIO(encoded_jpg)
                image = Image.open(encoded_jpg_io)
                img_encoded.append(encoded_jpg)
                imgs_name.append(img_file.name)
                imgs.append(image)

            st.session_state[key_1] = img_encoded
            st.session_state[key_2] = imgs_name
            st.session_state[key_3] = imgs

    def step_4_function(option):
        if option == "Train":
            key_1, key_2 = 'train', 'uploaded_img_train'
        elif option == "Test":
            key_1, key_2 = 'test', 'uploaded_img_test'

        st.session_state[key_2] = st.file_uploader(label="Upload image files from {} dataset.".format(key_1), 
                                                                type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

    def step_4_interface():
        st.info('**Step 4:** Upload the image files from train and test datasets.')
        col_upload_img_train, col_upload_img_test = st.columns(2)
        with col_upload_img_train:
            option = "Train"
            step_4_function(option)

        with col_upload_img_test:
            option = "Test"
            step_4_function(option)

    # Step 5. Generate TF Record
    def step_5_function():
        st.info('**Step 5:** Generate TF record for train and test datasets.')
        st.session_state['TF_Record_User_Selection'] = st.multiselect(label="Select to generate TF record file.", options=["Training dataset TF Record", "Testing Dataset TF record"])
    
    # Assemble all interface
    step_1_interface()
    step_2_interface()
    step_3_interface()
    step_4_interface()
    step_5_function()

    submit_button = st.button('Generate TF Record')
    if submit_button:
        # Step 1. Generate streamlit user directory
        if st.session_state["user_dir_choice"]=="+ Create a new directory":
            x_folder_name = st.session_state["user_create_dir_name"] + "/"
            st.session_state["streamlit_user_final_dir_choice"] = streamlit_user_directory + "/" + st.session_state["user_create_dir_name"] + "/"
            generate_training_dir(x_folder_name, x_folder_subdir)
        else: 
            st.session_state["streamlit_user_final_dir_choice"] = streamlit_user_directory + "/" + st.session_state["user_dir_choice"] + "/"

        # Step 2. Upload XML 
        xml_to_csv(st.session_state['xml_train'], option='Train')
        xml_to_csv(st.session_state['xml_test'], option='Test')

        # Step 3. Generate label map
        if st.session_state["user_dir_choice"]=="+ Create a new directory":
            dataset_filename = st.session_state["user_create_dir_name"]
        else: 
            dataset_filename = st.session_state["user_dir_choice"]

        label_map_output_path = st.session_state["streamlit_user_final_dir_choice"] + x_folder_subdir[0] + "/" + dataset_filename + "_label_map.pbtxt"
        if os.path.exists(label_map_output_path):
            pass
        else:
            generate_label_map(label_map_output_path)

        # Step 4. Upload images
        process_img(st.session_state['uploaded_img_train'], option='train')
        process_img(st.session_state['uploaded_img_test'], option='test')

        # Step 5. Generate TF record

        csv_output_path_train = st.session_state["streamlit_user_final_dir_choice"] + x_folder_subdir[0] + "/" + dataset_filename + "_train.csv"
        csv_output_path_test = st.session_state["streamlit_user_final_dir_choice"] + x_folder_subdir[0] + "/" + dataset_filename + "_test.csv"
        tf_record_output_path_train = st.session_state["streamlit_user_final_dir_choice"] + x_folder_subdir[0] + "/" + dataset_filename + "_train.record"
        tf_record_output_path_test = st.session_state["streamlit_user_final_dir_choice"] + x_folder_subdir[0] + "/" + dataset_filename + "_test.record"

        if not bool(st.session_state['TF_Record_User_Selection']):
            sys.exit("Please select which dataset to generate TF record in Step 5.")

        else:
            if len(st.session_state['TF_Record_User_Selection'])==1:
                if st.session_state['TF_Record_User_Selection']=="Training dataset TF Record":
                    create_tf_example(st.session_state['xml_df_train'], 
                                      st.session_state['image_train'], 
                                      st.session_state['imgs_name_train'], 
                                      st.session_state['img_encoded_train'], 
                                      csv_output_path_train, 
                                      tf_record_output_path_train, 
                                      label_map_output_path)
                    st.markdown("""
                    <section class="create_tf_record_warning_2" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                    <p>
                    Train TF record and CSV files have been generated. The directory is at:<br>
                    {}
                    </p>
                    </section>""".format(st.session_state["streamlit_user_final_dir_choice"] + "dataset"), unsafe_allow_html=True)
                    clear_session_state()
                else: # "Testing dataset TF Record"
                    create_tf_example(st.session_state['xml_df_test'], 
                                      st.session_state['image_test'], 
                                      st.session_state['imgs_name_test'], 
                                      st.session_state['img_encoded_test'], 
                                      csv_output_path_test, 
                                      tf_record_output_path_test, 
                                      label_map_output_path)
                    st.markdown("""
                    <section class="create_tf_record_warning_2" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                    <p>
                    Test TF record and CSV files have been generated. The directory is at:<br>
                    {}
                    </p>
                    </section>""".format(st.session_state["streamlit_user_final_dir_choice"] + "dataset"), unsafe_allow_html=True)
                    clear_session_state()
            else: # Both choices are selected

                create_tf_example(st.session_state['xml_df_train'], 
                                      st.session_state['image_train'], 
                                      st.session_state['imgs_name_train'], 
                                      st.session_state['img_encoded_train'], 
                                      csv_output_path_train, 
                                      tf_record_output_path_train, 
                                      label_map_output_path)
                create_tf_example(st.session_state['xml_df_test'], 
                                      st.session_state['image_test'], 
                                      st.session_state['imgs_name_test'], 
                                      st.session_state['img_encoded_test'], 
                                      csv_output_path_test, 
                                      tf_record_output_path_test, 
                                      label_map_output_path)
                st.markdown("""
                <section class="create_tf_record_warning_2" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                <p>
                Train and Test TF record and CSV files have been generated. The directory is at:<br>
                {}
                </p>
                </section>""".format(st.session_state["streamlit_user_final_dir_choice"] + "dataset"), unsafe_allow_html=True)
                clear_session_state()

        


