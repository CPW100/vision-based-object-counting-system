import streamlit as st
import sys
import io
import os
import argparse
import json

from PIL import Image
from labelme import utils
from slabelme2coco import labelme2coco
from screate_coco_tf_record import coco_tf_record

import numpy as np
import glob
import PIL.Image


def app():
    # Specify main directory
    main_dir = os.getcwd()
    split_path = main_dir.split("/")
    split_path = split_path[:len(split_path)-1]
    separator = "/"
    rejoin_path = separator.join(split_path)
    streamlit_user_directory = rejoin_path + "/" + "streamlit_user_directory"
    x_folder_subdir = ("dataset", "training", "fine_tuned", "pipeline", "evaluation", "node-red")

    # Streamlit Interface
    st.subheader("""Test Custom Object Dection Model""")

    # Initializing Session State
    session_state_keys = {"uploaded_json_file_train": None, 'uploaded_json_file_test': None, 'labels': None, 'number_of_label': None, 
                          'streamlit_user_dir_list': None,'empty_container_1': None, 'user_dir_choice': None, 'user_create_dir_name': None, 
                          'streamlit_user_final_dir_choice': None, 'uploaded_img_train': None, 'uploaded_img_test': None,
                          'img_encoded_train': None, 'img_encoded_test': None, 'imgs_name_train': None, 'imgs_name_test': None,
                          'TF_Record_User_Selection': None,}
    for key, value in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Supportive function
    def list_out_existing_user_directory(streamlit_user_directory):
            list_dir = []
            for filename in os.listdir(streamlit_user_directory):
                list_dir.append(filename)
            return list_dir

    def generate_training_dir(x_folder_name, x_folder_subdir):
            x_path = rejoin_path + '/streamlit_user_directory/' + x_folder_name
            if not os.path.exists(x_path):
                print("Generating training directory")
                create_directory(x_path)
                for item in x_folder_subdir:
                    create_directory(x_path + '/' + item)
                print(f"Training directory is at: {x_path}")

    def process_img(uploaded_image_files, option=None):
        if option=='train':
            key_1, key_2 = 'img_encoded_train', 'imgs_name_train'
        elif option=='test':
            key_1, key_2 = 'img_encoded_test', 'imgs_name_test'

        imgs_name, img_encoded = [], []
        if uploaded_image_files is not None:
            for img_file in uploaded_image_files:
                encoded_jpg = img_file.read()
                encoded_jpg_io = io.BytesIO(encoded_jpg)
                image = Image.open(encoded_jpg_io)
                img_encoded.append(encoded_jpg)
                imgs_name.append(img_file.name)

            st.session_state[key_1] = img_encoded
            st.session_state[key_2] = imgs_name

    def generate_label_map(label_map_output_path):
        if not os.path.exists(label_map_output_path):
            label_map_list = []
            labels = st.session_state['labels'].split(",")
            if st.session_state['labels']==None or st.session_state['labels']=="":
                sys.exit("Please enter image labelin step 2.")
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

    def generate_mask_tfrecord(json_abs_path, tfrecord_output_abspath, key_1, key_2, option=None):
        if not os.path.exists(json_abs_path):
            sys.exit("Please upload JSON file of {} dataset in Step 3.".format(option))

        if not bool(st.session_state[key_1]):
            sys.exit("Please upload image file of {} dataset in Step 4.".format(option))
        generate_coco_tf_record = coco_tf_record(json_abs_path,
                                                 tfrecord_output_abspath,
                                                 st.session_state[key_1],
                                                 st.session_state[key_2])
        generate_coco_tf_record.main()

    def clear_session_states():
        for key, value in session_state_keys.items():
                st.session_state[key] = None

    # Step 1. Select directory
    def select_directory_interface():
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
        if st.session_state["user_dir_choice"]=="+ Create a new directory":
            st.session_state["user_create_dir_name"] = st.session_state["empty_container_1"].text_input(label="Enter a name for the folder to be created.")
        else:
            st.session_state["empty_container_1"] = st.empty()
    
    # Step 2. Generate label map
    def label_map_interface():
        st.info('**Step 2:** Generate label map.')
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

    # Step 3. Upload JSON files
    def step_3_function(option):
        if option == "Train":
            key_1, key_2 = 'train', 'uploaded_json_file_train'
        elif option == "Test":
            key_1, key_2 = 'test', 'uploaded_json_file_test'

        st.session_state[key_2] = st.file_uploader(label="Upload JSON files from {} dataset.".format(key_1), 
                                                                type=['json'], accept_multiple_files=True)

    def upload_JSON_interface():
        st.info('**Step 3:** Upload the JSON files from train and test datasets.')
        col_upload_xml_train, col_upload_xml_test = st.columns(2)
        with col_upload_xml_train:
            option = "Train"
            step_3_function(option)

        with col_upload_xml_test:
            option = "Test"
            step_3_function(option)

    # Step 4. Upload image file
    def step_4_function(option):
        if option == "Train":
            key_1, key_2 = 'train', 'uploaded_img_train'
        elif option == "Test":
            key_1, key_2 = 'test', 'uploaded_img_test'

        st.session_state[key_2] = st.file_uploader(label="Upload image files from {} dataset.".format(key_1), 
                                                                type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)


    def upload_image_interface():
        st.info('**Step 4:** Upload the image files from train and test datasets.')
        col_upload_img_train, col_upload_img_test = st.columns(2)
        with col_upload_img_train:
            option = "Train"
            step_4_function(option)

        with col_upload_img_test:
            option = "Test"
            step_4_function(option)

    # Step 5. Generate train/test mask tf.record
    def generate_tfrecord_interface():
        st.info('**Step 5:** Generate TF record for train and test datasets labeled with mask.')
        st.session_state['TF_Record_User_Selection'] = st.multiselect(label="Select to generate TF record file.",
                                                                      options=["Training dataset TF Record", "Testing Dataset TF record"])

    # Assembling all interface and Run function
    select_directory_interface()
    label_map_interface()
    upload_JSON_interface()
    upload_image_interface()
    generate_tfrecord_interface()
    submit_button = st.button('Submit')
    if submit_button:
        # Step 1. Select/Create directory
        if st.session_state["user_create_dir_name"] is not None:
            x_folder_name = st.session_state["user_create_dir_name"] + "/"
            st.session_state["streamlit_user_final_dir_choice"] = streamlit_user_directory +  "/" + st.session_state["user_create_dir_name"] + "/"
            generate_training_dir(x_folder_name, x_folder_subdir)
        else:
            st.session_state["streamlit_user_final_dir_choice"] = streamlit_user_directory +  "/" + st.session_state["user_dir_choice"] + "/"

        # Step 2. Generate Label Map
        label_map_output_path = st.session_state["streamlit_user_final_dir_choice"] + x_folder_subdir[0] + "/" + st.session_state["user_dir_choice"] + "_label_map.pbtxt"
        if os.path.exists(label_map_output_path):
            pass
        else:
            generate_label_map(label_map_output_path)

        # Step 3. Call labelme2coco class
        path = st.session_state["streamlit_user_final_dir_choice"] + x_folder_subdir[0]
        if st.session_state['uploaded_json_file_train'] is not None: 
            abs_path_train = path + "/" + st.session_state["user_dir_choice"] + "_train.json"
            Labelme2Coco = labelme2coco(st.session_state['uploaded_json_file_train'], abs_path_train)
        if st.session_state['uploaded_json_file_test'] is not None: 
            abs_path_test = path + "/" + st.session_state["user_dir_choice"] + "_test.json"
            Labelme2Coco = labelme2coco(st.session_state['uploaded_json_file_test'], abs_path_test) 

        # Step 4. Generate tf.record
        process_img(st.session_state['uploaded_img_train'], option='train')
        process_img(st.session_state['uploaded_img_test'], option='test')
        output_abspath_train = path + "/" + st.session_state["user_dir_choice"] + "_mask_train.record"
        output_abspath_test = path + "/" + st.session_state["user_dir_choice"] + "_mask_test.record"
        len_option = len(st.session_state['TF_Record_User_Selection'])
        if not bool(st.session_state['TF_Record_User_Selection']):
            sys.exit("Please select which dataset to generate TF record in Step 5.")
        else:
            if len_option==1:
                if st.session_state['TF_Record_User_Selection'][-1]=='Training dataset TF Record':
                    generate_mask_tfrecord(abs_path_train, output_abspath_train, 'imgs_name_train', 'img_encoded_train', option='train')
                    st.markdown("""
                    <section class="create_tf_record_warning_2" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                    <p>
                    Train dataset TF Record is created and saved in folder:
                    {}
                    </p>
                    </section>""".format(path), unsafe_allow_html=True)
                    clear_session_states()

                elif st.session_state['TF_Record_User_Selection'][-1]=='Testing dataset TF Record':
                    generate_mask_tfrecord(abs_path_test, output_abspath_test, 'imgs_name_test', 'img_encoded_test', option='test')
                    st.markdown("""
                    <section class="create_tf_record_warning_2" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                    <p>
                    Test dataset TF Record is created and saved in folder:
                    {}
                    </p>
                    </section>""".format(path), unsafe_allow_html=True)
                    clear_session_states()
            else:
                generate_mask_tfrecord(abs_path_train, output_abspath_train, 'imgs_name_train', 'img_encoded_train', option='train')
                generate_mask_tfrecord(abs_path_test, output_abspath_test, 'imgs_name_test', 'img_encoded_test', option='test')
                st.markdown("""
                    <section class="create_tf_record_warning_2" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                    <p>
                    Train and Test dataset TF Records are created and saved in folder:
                    {}
                    </p>
                    </section>""".format(path), unsafe_allow_html=True)
                clear_session_states()
        clear_session_states()


        



