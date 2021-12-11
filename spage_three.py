import streamlit as st
import glob
import requests
import sys
import os
import tarfile
import shutil
from stqdm import stqdm



def app():
    main_dir = os.getcwd()
    split_path = main_dir.split("/")
    split_path = split_path[:len(split_path)-1]
    separator = "/"
    rejoin_path = separator.join(split_path)
    streamlit_user_directory = rejoin_path + "/" + "streamlit_user_directory"
    x_folder_subdir = ("dataset", "training", "fine_tuned", "pipeline", "evaluation", "node-red")
    pretrained_model_path = rejoin_path + "/" + "streamlit_pretrained_models"
    model = {
            "Faster-RCNN-ResNet-50": {
                "pretrained_checkpoint": "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
                },

            "Mask-RCNN": {
                "pretrained_checkpoint": "mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
                },

            "SSD-EfficientDet": {
                "pretrained_checkpoint": "efficientdet_d0_coco17_tpu-32.tar.gz"
                                 }
        }

    pretrained_model = {
            "Faster-RCNN-ResNet-50": "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/checkpoint/ckpt-0.index" ,
            "Mask-RCNN": "mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/checkpoint/ckpt-0.index",
            "SSD-EfficientDet": "efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0.index"
        }

    # Initializing Session State
    session_state_keys = {"pretrained_model_listdir": None, "user_chosen_model": None, "user_chosen_subdir_1": None, "user_chosen_subdir_2": None,
                          "user_chosen_subdir_3": None, "user_chosen_subdir_4": None, "user_chosen_subdir_4": None, "user_chosen_dir": None, 
                          "subdir_fpath_list": [], "file_or_folder": None}
    for key, value in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Create pre-trained model directory
    if not os.path.exists(pretrained_model_path): 
        os.mkdir(pretrained_model_path)

    """Downloading functions"""
    # 1. Download pretrained model
    def Download_pretrained_model(tar_path, pretrained_checkpoint,chosen_model, pretrained_model_ckpt_fpath):
        assert tar_path, "Please provide a correct path for pretrained model to be downloaded."
        download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'  + pretrained_checkpoint
        # The header of the dl link has a Content-Length which is in bytes.
        # The bytes is in string hence has to convert to integer.
        filesize = int(requests.head(download_tar).headers["Content-Length"])
        filename = pretrained_checkpoint
        chunk_size = 1024
        output_path = tar_path + '/' + chosen_model
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        print("Pretrained Model Download Status: Starting to download...")
        print(f"Pretrained Model Download Path: {os.path.join(output_path, pretrained_checkpoint)}")
        with requests.get(download_tar, stream=True) as r, open(output_path + '/' + pretrained_checkpoint, "wb") as f, stqdm(
        unit="B",  # unit string to be displayed.
        unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
        unit_divisor=1024,  # is used when unit_scale is true
        total=filesize,  # the total iteration.
        file=sys.stdout,  # default goes to stderr, this is the display on console.
        desc=filename,  # prefix to be displayed on progress bar.
        backend=True
        ) as progress:
            for chunk in r.iter_content(chunk_size=chunk_size):
                # download the file chunk by chunk
                datasize = f.write(chunk)
                # on each chunk update the progress bar.
                progress.update(datasize)

        #tar_response = requests.get(download_tar, stream=True)
        print("Pretrained Model Download Status: Extracting tar file.")
        #file = tarfile.open(fileobj=tar_response.raw, mode="r|gz")
        file = tarfile.open(output_path + '/' + pretrained_checkpoint, mode="r|gz")
        file.extractall(path=output_path)
        file.close()
        print("Pretrained Model Download Status: Download completed.")
        os.remove(output_path + '/' + pretrained_checkpoint)

    """Delete folder content or specific file function"""
    def delete_file(path):
        if os.path.isdir(path):
            folder = path
            try:
                shutil.rmtree(path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (path, e))
        elif os.path.isfile(path):
            try:
                os.remove(path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (path, e))

        else:
            try:
                pass
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (path, e))

    def list_out_existing_user_directory(streamlit_user_directory):
            list_dir = []
            for filename in os.listdir(streamlit_user_directory):
                list_dir.append(filename)
            return list_dir

    # 2. Download Tensorflow Models Repository

    """Interface Function"""
    # 1. Download pretrained model interface
    def DL_pretrained_model_interface():
        st.session_state["user_chosen_model"] = st.selectbox(label="Please select which pretrained model to be downloaded.",
                                                             options=["Faster-RCNN-ResNet-50", "Mask-RCNN",  "SSD-EfficientDet"], help="")
        download_pretrained_model_button = st.button(label="Download Pretrained Model")
        if download_pretrained_model_button:
            pretrained_model_file_exists = False
            for file in os.listdir(pretrained_model_path):
                pretrained_model_ckpt_fpath = pretrained_model_path + "/" + st.session_state["user_chosen_model"] + "/" + pretrained_model[st.session_state["user_chosen_model"]]
                if os.path.exists(pretrained_model_ckpt_fpath):
                    st.markdown("""
                        <section class="create_tf_record_warning_1" style="background-color:#ECD4FF; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                        <p>
                        The {} pretrained model already exists.
                        </p>
                        </section>
                        <p></p>""".format(st.session_state["user_chosen_model"]), unsafe_allow_html=True)
                    pretrained_model_file_exists = True

            if pretrained_model_file_exists == False:
                chosen_model = st.session_state["user_chosen_model"]
                pretrained_checkpoint = model[st.session_state["user_chosen_model"]]["pretrained_checkpoint"]
                Download_pretrained_model(pretrained_model_path, pretrained_checkpoint,chosen_model, pretrained_model_ckpt_fpath)
                st.markdown("""
                <section class="create_tf_record_warning_1" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                <p>
                The {} pretrained model has been downloaded.
                </p>
                </section>
                <p></p>""".format(st.session_state["user_chosen_model"]), unsafe_allow_html=True)

    def delete_file_interface():
        
        user_main_dir_namelist = os.listdir(streamlit_user_directory)
        if user_main_dir_namelist:
            st.session_state["user_chosen_dir"] = st.selectbox(label="Select a directory.", options=user_main_dir_namelist, )
            subdir_col_1, subdir_col_2 = st.columns(2)
            subdir_col_3, subdir_col_4 = st.columns(2)
            with subdir_col_1:
                st.info("Sub-directory 1: Dataset")
                option_1 = list_out_existing_user_directory(streamlit_user_directory + "/" + st.session_state["user_chosen_dir"] + "/" + x_folder_subdir[0])
                
                if not bool(option_1):
                    st.warning("There is no file in this sub-directory.")
                    st.session_state["user_chosen_subdir_1"] = "Not chosen"
                else:
                    option_1.append("Not chosen")
                    st.session_state["user_chosen_subdir_1"] = st.multiselect("Select your choice of item in sub-directory 1.", options=option_1, default=option_1[-1])

            with subdir_col_2:
                st.info("Sub-directory 2: Training Model")
                option_2 = list_out_existing_user_directory(streamlit_user_directory + "/" + st.session_state["user_chosen_dir"] + "/" + x_folder_subdir[1])
                if not bool(option_2):
                    st.warning("There is no file in this sub-directory.")
                    st.session_state["user_chosen_subdir_2"] = "Not chosen"
                else:
                    option_2.append("Not chosen")
                    st.session_state["user_chosen_subdir_2"] = st.multiselect(label="Select your choice of item in sub-directory 2.", options=option_2, default=option_2[-1])

            with subdir_col_3:
                st.info("Sub-directory 3: Fine Tuned Model") 
                option_3 = list_out_existing_user_directory(streamlit_user_directory + "/" + st.session_state["user_chosen_dir"] + "/" + x_folder_subdir[2])
                if not bool(option_3):
                    st.warning("There is no file in this sub-directory.")
                    st.session_state["user_chosen_subdir_3"] = "Not chosen"
                else:
                    option_3.append("Not chosen")
                    st.session_state["user_chosen_subdir_3"] = st.multiselect(label="Select your choice of item in sub-directory 3.", options=option_3, default=option_3[-1])

            with subdir_col_4:
                st.info("Sub-directory 4: Pipeline")
                option_4 = list_out_existing_user_directory(streamlit_user_directory + "/" + st.session_state["user_chosen_dir"] + "/" + x_folder_subdir[3])
                if not bool(option_4):
                    st.warning("There is no file in this sub-directory.")
                    st.session_state["user_chosen_subdir_4"] = "Not chosen"
                else:
                    option_4.append("Not chosen")
                    st.session_state["user_chosen_subdir_4"] = st.multiselect(label="Select your choice of item in sub-directory 4.", options=option_4, default=option_4[-1])

            st.info("Sub-directory 5: Evaluated Model")
            option_5 = list_out_existing_user_directory(streamlit_user_directory + "/" + st.session_state["user_chosen_dir"] + "/" + x_folder_subdir[4])
            if not bool(option_5):
                st.warning("There is no file in this sub-directory.")
                st.session_state["user_chosen_subdir_5"] = "Not chosen"
            else:
                option_5.append("Not chosen")
                st.session_state["user_chosen_subdir_5"] = st.multiselect("Select your choice of item in sub-directory 5.", options=option_5, default=option_5[-1])

            subdir_list = [st.session_state["user_chosen_subdir_1"], st.session_state["user_chosen_subdir_2"], st.session_state["user_chosen_subdir_3"], st.session_state["user_chosen_subdir_4"], st.session_state["user_chosen_subdir_5"]] 
            st.session_state["subdir_fpath_list"] = []
            for i, item in enumerate(subdir_list):
                if not isinstance(item, list):
                    if not item == "Not chosen":
                        chosen_fpath = streamlit_user_directory + "/" + st.session_state["user_chosen_dir"] + "/" + x_folder_subdir[i] + "/" + item
                        
                        st.session_state["subdir_fpath_list"].append(chosen_fpath)
                else:
                    for sub_item in item:
                        if not sub_item == "Not chosen":
                            chosen_fpath = streamlit_user_directory + "/" + st.session_state["user_chosen_dir"] + "/" + x_folder_subdir[i] + "/" + sub_item
                            st.session_state["subdir_fpath_list"].append(chosen_fpath)
            st.session_state["file_or_folder"] = st.selectbox(label="Would you like to delete a directory of specific file?", options=["directory", "specific file from sub-directory"])
            delete_file_button = st.button("Delete Chosen File")
            if delete_file_button:
                if st.session_state["file_or_folder"]=="directory":
                    delete_file(path=streamlit_user_directory + "/" + st.session_state["user_chosen_dir"])
                    st.markdown("""
                    <section class="create_tf_record_warning_1" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                    <p>
                    The <b>{}</b> directory has been deleted.
                    </p>
                    </section>
                    <p></p>""".format(st.session_state["user_chosen_dir"]), unsafe_allow_html=True)

                elif st.session_state["file_or_folder"]=="specific file from sub-directory":
                    for item_to_be_deleted in st.session_state["subdir_fpath_list"]:
                        delete_file(path=item_to_be_deleted)
                    st.markdown("""
                    <section class="create_tf_record_warning_1" style="background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                    <p>
                    Deleted file list:<br>
                    {}
                    </p>
                    </section>
                    <p></p>""".format(st.session_state["subdir_fpath_list"]), unsafe_allow_html=True)
            
        else:
            st.markdown("""
                <section class="create_tf_record_warning_1" style="background-color:#ECD4FF; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                <p>
                There is no custom directory ceated yet. You may navigate to <b><i>Upload Data</i></b> page to create one.
                </p>
                </section>
                <p></p>""", unsafe_allow_html=True)

    # Streamlit Interface
    st.subheader("""User File Manager""")
    with st.expander(label="Download Tensorflow pretrained Model"):
        DL_pretrained_model_interface()
    with st.expander(label="Manage User Directory"):
        delete_file_interface()

    

    