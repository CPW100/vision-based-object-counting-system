import streamlit as st
import os
import shutil
import glob
import regex as re
import tensorflow as tf

# custom import
import train_evaluate_model
import sexporter_main_v2 as exporter_main_v2

from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def app():
    # Specify main directory
    main_dir = os.getcwd()
    split_path = main_dir.split("/")
    split_path = split_path[:len(split_path)-1]
    separator = "/"
    rejoin_path = separator.join(split_path)
    streamlit_user_directory = rejoin_path + "/" + "streamlit_user_directory"
    streamlit_pretrained_direcory = rejoin_path + "/" + "streamlit_pretrained_models"
    x_folder_subdir = ("dataset", "training", "fine_tuned", "pipeline", "evaluation", "node-red")

    # Streamlit Interface
    st.subheader("""Train Custom Object Dection Model""")

    # Initializing Session State
    session_state_keys = {"training_dir": None, "custom_pipeline_config": None, "num_train_steps": 5000, "sample_1_of_n_eval_examples": 1, "chosen_streamlit_directory": None,
                          "custom_pipeline_config_path": None, "user_chosen_model": None, "fine_tuned_dir": None, "train_choice_1": None, "train_choice_2": None,
                          "previous_ckpt": None,}
    for key, value in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value

    pretrained_model = {
                "Faster-RCNN-ResNet-50": "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/checkpoint/ckpt-0" ,
                "Mask-RCNN": "mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/checkpoint/ckpt-0",
                "SSD-EfficientDet": "efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0"
            }

    """Model training user interface"""
    def list_out_existing_user_directory(streamlit_user_directory):
        list_dir = []
        for filename in os.listdir(streamlit_user_directory):
            list_dir.append(filename)
        return list_dir

    def train_my_model():
        train_model = train_evaluate_model.train_or_evaluate_model(model_dir = st.session_state["training_dir"], pipeline_config_path = st.session_state["custom_pipeline_config_path"])
        train_model.num_train_steps = st.session_state["num_train_steps"]
        train_model.sample_1_of_n_eval_examples= st.session_state["sample_1_of_n_eval_examples"]
        train_model.start()
        export_finetunedmodel = exporter_main_v2.export_fine_tuned_model()
        export_finetunedmodel.pipeline_config_path = st.session_state["custom_pipeline_config_path"]
        export_finetunedmodel.trained_checkpoint_dir = st.session_state["training_dir"]
        export_finetunedmodel.output_directory = st.session_state["fine_tuned_dir"]
        export_finetunedmodel.run()

    def delete_all_files(path_to_folder):
        folder = path_to_folder
        for filename in os.listdir(folder):
            if filename:
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    def modify_pipeline_ckpt(previous_ckpt_list, option=True):

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(st.session_state["custom_pipeline_config_path"], "r") as f:                                                                                                                                                                                                                     
            proto_str = f.read()                                                                                                                                                                                                                                          
            text_format.Merge(proto_str, pipeline_config)

        if option==True:
            pipeline_config.train_config.fine_tune_checkpoint =  st.session_state["training_dir"] + "/" + previous_ckpt_list[-1].split(".")[0] 
            config_text = text_format.MessageToString(pipeline_config)  
            with tf.io.gfile.GFile(st.session_state["custom_pipeline_config_path"], "wb") as f:                                                                                                                                                                                                                     
                f.write(config_text)

        elif option==False:
            pipeline_config.train_config.fine_tune_checkpoint = streamlit_pretrained_direcory + "/" + st.session_state["user_chosen_model"]  + "/" + pretrained_model[st.session_state["user_chosen_model"]]
            config_text = text_format.MessageToString(pipeline_config)  
            with tf.io.gfile.GFile(st.session_state["custom_pipeline_config_path"], "wb") as f:                                                                                                                                                                                                                     
                f.write(config_text)

    def training_interface():
        streamlit_user_dir_list = list_out_existing_user_directory(streamlit_user_directory)
        st.info("""**Step 1:** Select custom dataset to train an object detection model.""")
        if bool(streamlit_user_dir_list):
            st.session_state["chosen_streamlit_directory"] = st.selectbox(label="Please select which custom dataset you would like to train the model on.",
                                                                          options=streamlit_user_dir_list)
            st.markdown("""
            <section class="create_tf_record_warning_2" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
            <p>
            The chosen custom dataset is <b>{}</b>
            </p>
            </section>
            <p></p>""".format(st.session_state["chosen_streamlit_directory"]), unsafe_allow_html=True)
        else:
            st.markdown("""
            <section class="create_tf_record_warning_2" style="background-color:#ECD4FF; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
            <p>
            There is no custom directory created yet.<br>
            Please navigate to <b><i>Upload Data</i></b> page to create one. 
            </p>
            </section>
            <p></p>""", unsafe_allow_html=True)
        streamlit_user_pipeline_directory = streamlit_user_directory + "/" + st.session_state["chosen_streamlit_directory"] + "/" + x_folder_subdir[3]
        streamlit_user_fine_tuned_directory = streamlit_user_directory + "/" + st.session_state["chosen_streamlit_directory"] + "/" + x_folder_subdir[2]
        print(f"streamlit_user_pipeline_directory: {streamlit_user_pipeline_directory}")

        st.info("""**Step 2:** Select custom pipeline configuration to train an object detection model.""")
        streamlit_user_pipeline_list = []
        for config_file in glob.glob(streamlit_user_pipeline_directory + "/*.config"):
            chosen_config_file = config_file.split(".")[0]
            chosen_config_file = chosen_config_file.split("/")[-1]
            streamlit_user_pipeline_list.append(chosen_config_file)
        st.session_state["custom_pipeline_config"] = st.selectbox(label="Please select which custom model pipeline you would like to train the model on.",
                                                                options=streamlit_user_pipeline_list)
        if st.session_state["custom_pipeline_config"] is not None or "":
            st.session_state["custom_pipeline_config_path"] = streamlit_user_pipeline_directory + "/" + st.session_state["custom_pipeline_config"] + ".config"
            st.session_state["user_chosen_model"] = st.session_state["custom_pipeline_config"].split("_")[0]
            st.markdown("""
            <section class="create_tf_record_warning_2" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
            <p>
            The chosen model configuration is <b>{}</b>
            </p>
            </section>
            <p></p>""".format(st.session_state["user_chosen_model"]), unsafe_allow_html=True)          
        else:
            st.markdown("""
            <section class="create_tf_record_warning_2" style="background-color:#ECD4FF; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
            <p>
            There is no custom model pipeline created yet.<br>
            Please navigate to <b><i>Configure Model Pipeline</i></b> page to create one. 
            </p>
            </section>
            <p></p>""", unsafe_allow_html=True)
        
        if st.session_state["custom_pipeline_config"] is not None:
            st.session_state["training_dir"] =  streamlit_user_directory + "/" + st.session_state["chosen_streamlit_directory"] + "/" + x_folder_subdir[1] + "/" + st.session_state["custom_pipeline_config"]
            st.session_state["fine_tuned_dir"] = streamlit_user_fine_tuned_directory + "/" + st.session_state["custom_pipeline_config"]

        st.info('**Step 3:** Set the number of steps for the model that is to be trained.')
        st.session_state["num_train_steps"] = st.slider(label="Set number of training steps.",
                                                                    min_value=1000, max_value=100000, value=5000, step=1000)
        st.markdown("""
        <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
        <p>
        The number of training step is <b>{}</b>.  
        </p>
        </section>
        <p></p>
        <p></p>
        """.format(st.session_state["num_train_steps"]), unsafe_allow_html=True)

        st.info("**Step 4:** Set 'X' number of test dataset examples where the model will be evaluated by one of every 'X' test input dataset.")
        st.session_state["sample_1_of_n_eval_examples"] = st.slider(label="Set number of 'X'", min_value=1, max_value=20, value=5, step=1)
        st.markdown("""
        <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
        <p>
        During training process, model will be evaluated by one of every <b>{}</b> test input dataset.  
        </p>
        </section>
        <p></p>
        <p></p>
        """.format(st.session_state["sample_1_of_n_eval_examples"]), unsafe_allow_html=True)


        #st.info("**Step 5:**Choose whether or not to continue training with previous checkpoint or train from scratch.")
        #st.session_state["previous_ckpt"] = False
        #previous_ckpt_list = []
        # Check if there is any existing checkpoint trained previously.

        #for ckpt_filename in os.listdir(st.session_state["training_dir"]):
        #    if re.match(r'\w+\-\w+[.]index', ckpt_filename):
        #        previous_ckpt_list.append(ckpt_filename)
        #        st.session_state["previous_ckpt"] = True

        #number_of_previous_ckpt = len(previous_ckpt_list)
        #if st.session_state["previous_ckpt"] == True:
        #    if number_of_previous_ckpt>=1:
        #        if number_of_previous_ckpt==1:
        #            train_label = "There is an existng checkpoint trained previously, would you like to overwrite it?"
        #        else:
        #            train_label = f"There are {number_of_previous_ckpt} existng checkpoints trained previously, would you like to overwrite it?"
        #        q1_col, q2_col = st.columns(2)    
        #        with q1_col:
        #            st.session_state["train_choice_1"] = st.radio(label=train_label, options=["No", "Yes"])
        #            if st.session_state["train_choice_1"]=="No":
        #                with q2_col():
        #                    st.session_state["train_choice_2"] = st.radio(label="If so, would you like to continue training using the previous checkpoint?", options=["No", "Yes"])
        #        if  st.session_state["train_choice_1"]=="No" and st.session_state["train_choice_2"]=="No":
        #            modify_pipeline_ckpt(previous_ckpt_list, option=False)
        #            st.markdown("""
        #            <section style="background-color:#FFA726; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
        #            <p>
        #            Please select or create a new model configuration to train your custom model.<br>
        #            You can navigate to <b><i>Configure Model Pipeline</i></b> page configure a new model.
        #            </p>
        #            </section>
        #            <p></p>""".format(number_of_previous_ckpt-1), unsafe_allow_html=True)

        #        elif  st.session_state["train_choice_1"]=="No" and st.session_state["train_choice_2"]=="Yes":
        #            modify_pipeline_ckpt(previous_ckpt_list, option=True)
        #            st.markdown("""
        #                <section style="background-color:#FFA726; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
        #                <p>
        #                Training process will continue from the previous checkpoint which is ckpt-{}.
        #                </p>
        #                </section>
        #                <p></p>""".format(number_of_previous_ckpt-1), unsafe_allow_html=True)

        #        elif st.session_state["train_choice_1"]=="Yes" :
        #            modify_pipeline_ckpt(previous_ckpt_list, option=False)
        #            delete_all_files(st.session_state["training_dir"])
        #            delete_all_files(st.session_state["fine_tuned_dir"])
        #            st.markdown("""
        #            <section class="create_tf_record_warning_2" style="background-color:#FFA726; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
        #            <p>
        #            Training process will start from checkpoint 0.
        #            Previous checkpoints will be overwritten by new train checkpoints.
        #            </p>
        #            </section>
        #            <p></p>""", unsafe_allow_html=True)
        #elif st.session_state["previous_ckpt"]==False:
        #    modify_pipeline_ckpt(previous_ckpt_list, option=False)
        #    st.markdown("""
        #    <section class="create_tf_record_warning_2" style="background-color:#FFA726; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
        #    <p>
        #    No previous checkpoint exists. Training process will start from checkpoint 0.
        #    </p>
        #    </section>
        #    <p></p>""", unsafe_allow_html=True)

        train_button = st.button("Start Training My Model")
        
        if train_button:
            #if st.session_state["train_choice_2"]=="Yes" or st.session_state["previous_ckpt"]==False or st.session_state["train_choice_1"]=="Yes":
             train_my_model()
                #st.write("Train My Model :)")
    # Execute model training interface
    training_interface()
        
        
