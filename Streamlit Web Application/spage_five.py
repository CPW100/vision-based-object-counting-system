import os
import io
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# Custom import
from scounting_object import Evaluate_and_Count

def app():
    # Specify main directory
    main_dir = os.getcwd()
    split_path = main_dir.split("/")
    split_path = split_path[:len(split_path)-1]
    separator = "/"
    rejoin_path = separator.join(split_path)
    streamlit_user_directory = rejoin_path + "/" + "streamlit_user_directory"
    x_folder_subdir = ("dataset", "training", "fine_tuned", "pipeline", "evaluation", "node-red")
    key_list = []

    # Streamlit Interface
    st.subheader("""Test Custom Object Dection Model""")

    # Initializing Session State
    session_state_keys = {"training_dir": None, "custom_pipeline_config_path": None, "test_dataset_csv_fpath": None, "test_dataset_img_fpath": None,
                          "evaluation_img_fpath": None, "evaluation_csv_path": None, "chosen_streamlit_directory": None, "chosen_config_file": None,
                          "test_img": None, "eval_fpath": None, "test_img_name": None,}
    for key, value in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Interface function
    def clear_session_state():
        for key, value in session_state_keys.items():
            st.session_state[key] = None

    def list_out_existing_user_directory(streamlit_user_directory):
        list_dir = []
        for filename in os.listdir(streamlit_user_directory):
            list_dir.append(filename)
        return list_dir

    def evaluate_interface():
        streamlit_user_dir_list = list_out_existing_user_directory(streamlit_user_directory)

        st.info("""**Step 1:** Select a directory to save the evaluation results.""")
        if bool(streamlit_user_dir_list):
            st.session_state["chosen_streamlit_directory"] = st.selectbox(label="Choose a directory.", options=streamlit_user_dir_list)
            st.markdown("""
            <section style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
            <p>
            The chosen custom directory is <b>{}</b>
            </p>
            </section>
            <p></p>""".format(st.session_state["chosen_streamlit_directory"]), unsafe_allow_html=True)
            streamlit_user_config_directory = streamlit_user_directory + "/" + st.session_state["chosen_streamlit_directory"] + "/" + x_folder_subdir[3]
            streamlit_user_configfile_list = list_out_existing_user_directory(streamlit_user_config_directory)
            streamlit_user_configname_list = []
            for config_file in streamlit_user_configfile_list:
                streamlit_user_configname_list.append(config_file.split('.')[0])

            st.info("""**Step 2:** Select a directory to save the evaluation results.""")
            if bool(streamlit_user_configname_list):
                st.session_state["chosen_config_file"] = st.selectbox(label="Choose a directory.",options=streamlit_user_configname_list)
                st.markdown("""
                <section style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                <p>
                The chosen custom model configuraion is <b>{}</b>
                </p>
                </section>
                <p></p>""".format(st.session_state["chosen_config_file"]), unsafe_allow_html=True)
            else:
                st.markdown("""
                <section style="background-color:#ECD4FF; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                <p>
                You have not created any custom model yet.
                Please navigate to <i><b>Configure Model Pipeline</b></i> page to create one.
                </p>
                </section>
                <p></p>""", unsafe_allow_html=True)

            st.info("""**Step 3:** Select images from your test dataset which has been labeled to evaluate your custom object detection model.""")
            test_img = st.file_uploader(label="Upload image files from test dataset.", 
                                                    type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
            st.session_state["test_img"], st.session_state["test_img_name"]  = [], []
            if bool(test_img):
                i=0
                for i, img_file in enumerate(test_img):
                    encoded_jpg = img_file.read()
                    encoded_jpg_io = io.BytesIO(encoded_jpg)
                    image = Image.open(encoded_jpg_io)
                    st.session_state["test_img_name"].append(img_file.name)
                    st.session_state["test_img"].append(image)

                st.markdown("""
                <section style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                <p>
                Number of uploaded test image: {}
                </p>
                </section>
                <p></p>""".format(i+1), unsafe_allow_html=True)
            else:
                st.markdown("""
                <section style="background-color:#ECD4FF; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                <p>
                No images from test dataset has been uploaded yet.
                </p>
                </section>
                <p></p>""", unsafe_allow_html=True)

            submit_button = st.button("Evaluate Custom Model")
        else: 
            st.markdown("""
            <section style="background-color:#ECD4FF; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
            <p>
            There is no user directory created yet.
            Please navigate to <i><b>Upload Dataset</b></i> page to create one.
            </p>
            </section>
            <p></p>""", unsafe_allow_html=True)
        
        

        if submit_button:
            # Set value
            st.session_state["test_dataset_csv_fpath"] = streamlit_user_directory + "/" + st.session_state["chosen_streamlit_directory"] + "/" + x_folder_subdir[0] + "/" +  st.session_state["chosen_streamlit_directory"] + "_test.csv"
            st.session_state["custom_pipeline_config_path"] = streamlit_user_directory + "/" + st.session_state["chosen_streamlit_directory"] + "/" + x_folder_subdir[3] + "/" + st.session_state["chosen_config_file"] + ".config"
            st.session_state["training_dir"] = streamlit_user_directory + "/" + st.session_state["chosen_streamlit_directory"] + "/" + x_folder_subdir[1] + "/" + st.session_state["chosen_config_file"]
            st.session_state["eval_fpath"] = streamlit_user_directory + "/" + st.session_state["chosen_streamlit_directory"] + "/" + x_folder_subdir[4] + "/" + st.session_state["chosen_config_file"]
            st.session_state["evaluation_img_fpath"] = st.session_state["eval_fpath"] + "/" + "evaluated_img" 
            st.session_state["evaluation_csv_path"] = st.session_state["eval_fpath"] + "/" + "evaluated_csv_result"  
            if not os.path.isdir(st.session_state["eval_fpath"]):
                os.mkdir(st.session_state["eval_fpath"])
            if not os.path.isdir(st.session_state["evaluation_img_fpath"]):
                os.mkdir(st.session_state["evaluation_img_fpath"])
                os.mkdir(st.session_state["evaluation_csv_path"])
            
            # Calling Counting Class
            # test_dataset_csv_fpath, custom_pipeline_config_path, training_dir, evaluation_img_fpath, evaluation_csv_path, test_img
            eval_and_count = Evaluate_and_Count(st.session_state["test_dataset_csv_fpath"],
                                                st.session_state["custom_pipeline_config_path"],
                                                st.session_state["training_dir"],
                                                st.session_state["evaluation_img_fpath"],
                                                st.session_state["evaluation_csv_path"],
                                                st.session_state["test_img"],
                                                st.session_state["test_img_name"])
            eval_and_count.start()
            st.markdown("""
            <section style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
            <p>
            Evaluation process is completed. The evaluation result is at:<br>
            {}
            </p>
            </section>
            <p></p>""".format(st.session_state["eval_fpath"]), unsafe_allow_html=True)
            clear_session_state()

    # Assemble interface function
    evaluate_interface()
    


    