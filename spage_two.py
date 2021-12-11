import streamlit as st
import os
import glob

# Custom import 
from configure_pipeline import Pipeline_Config

r"""
The interface of spage_two.py is used to configure model structure.
"""
def app():
    """ Define User Directory"""
    main_dir = os.getcwd()
    split_path = main_dir.split("/")
    split_path = split_path[:len(split_path)-1]
    separator = "/"
    rejoin_path = separator.join(split_path)
    pipeline_reference_path = rejoin_path + "/" + "models/research/object_detection/configs/tf2"
    print(rejoin_path)
    print(os.path.isdir(rejoin_path))
    x_folder_subdir = ("dataset", "training", "fine_tuned", "pipeline", "evaluation", "node-red")
    model = {
                "Faster-RCNN-ResNet-50": "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config" ,
                "Mask-RCNN": "mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config",
                "SSD-EfficientDet": "ssd_efficientdet_d0_512x512_coco17_tpu-8.config"
            }
    pretrained_model = {
                "Faster-RCNN-ResNet-50": "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/checkpoint/ckpt-0" ,
                "Mask-RCNN": "mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/checkpoint/ckpt-0",
                "SSD-EfficientDet": "efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0"
            }
    model_dict = {
                1: "Faster-RCNN-ResNet-50",
                2: "Mask-RCNN",
                3: "SSD-EfficientDet"
                }

    optimizer_dict =  {
                        "momentum_optimizer": "Momentum Optimizer",
                        "adam_optimizer": "Adam Optimizer"
                      }
    # Streamlit Interface
    st.subheader("""Configure Model Parameter""")

    # Get ConfigPipe class
    configure_parameter = Pipeline_Config()

    # Initializing Session State --> st.session_state[""]
    session_state_keys = {"user_dir_choice": None, "streamlit_user_dir_list": None, "streamlit_user_final_dir_choice": None,
                          "min_img_dimension_config": 200, "max_img_dimension_config": 400, "tuple_img_dimension_config": None,
                          "empty_container_8": None, "num_steps": 5000, "batch_size": 4, "dropout": "false", "dropout_keep_prob": 1.00,
                          "optimizer": "Momentum", "warmup_learning_rate": 0.0133, "warmup_steps": 1000, "learning_rate_base": 0.04,
                          "total_steps": 5000, "random_horizontal_flip": None, "random_adjust_hue": None, "random_adjust_contrast": None,
                          "random_adjust_saturation": None, "random_square_crop_by_scale": None, "min_scale_1": 0.6, "max_scale_1": 1.3,
                          "data_augmentation": [], "data_aug_status": None, "iou_thres":0.8, "min_scale_2": 0.1, "max_scale_2": 2.0,
                          "output_size": 500, "model":None, "model_pipeline_path": None, "streamlit_user_model_count":0}
    for key, value in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 1. Choose model
    def step_1_interface():
        with st.expander(label="Step 1: Select model"):
            st.info('**Step 1:** Select a model to be trained on the uploaded custom dataset.')
            st.session_state["model"] = st.selectbox(label="Choose a model type.", options=[1,2,3], format_func=lambda option: model_dict[option], help="")
            st.session_state["model_pipeline_path"] = pipeline_reference_path + "/" + model[model_dict[st.session_state["model"]]]
            st.markdown("""
            <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
            <p>
            The chosen model is <b>{}</b>.  
            </p>
            </section>
            <p></p>
            <p></p>
            """.format(model_dict[st.session_state["model"]]), unsafe_allow_html=True)

    
    # 2. Change number of steps
    def step_2_interface():
        with st.expander(label="Step 2: Set number of training steps"):
            st.info('**Step 2:** Set the number of steps for the model that is to be trained.')
            st.session_state["num_steps"] = st.slider(label="Set number of training steps.",
                                                                        min_value=1000, max_value=100000, value=5000, step=1000)
            st.markdown("""
            <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
            <p>
            The number of training step is <b>{}</b>.  
            </p>
            </section>
            <p></p>
            <p></p>
            """.format(st.session_state["num_steps"]), unsafe_allow_html=True)

    # 3. Change IOU
    def step_3_interface():
        with st.expander(label='Step 3: Intersection Of Union Probability (IOU) (Optional)'):
            st.info("**Step 3:** Select a value for Intersection Of Union Probability.")
            st.session_state["iou_thres"] = st.slider(label="Select a value.", min_value=0.01, max_value=1.00, value=0.80, step=0.01)
            st.markdown("""
            <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
            <p>
            The IOU is set to <b>{}</b>
            </p>
            </section>
            <p></p>
            """.format(st.session_state["iou_thres"]), unsafe_allow_html=True) 

    # 4. Change Image Augmentation
    def step_4_function_1():
        st.session_state["min_scale_1"] = st.slider(label="Minimum scale", min_value=0.0, max_value=100.0, value=0.6, step=0.1, help="")
        st.session_state["max_scale_1"] = st.slider(label="Maximum scale", min_value=0.0, max_value=100.0, value=1.3, step=0.1, help="")

    def step_4_function_2():
        st.session_state["output_size"] = st.text_input(label="Enter the square image output size after cropping.", value=512,  help="")
        st.session_state["min_scale_2"] = st.slider(label="Minimum scale", min_value=0.0, max_value=100.0, value=0.1, step=0.1, help="")
        st.session_state["max_scale_2"] = st.slider(label="Maximum scale", min_value=0.0, max_value=100.0, value=2.0, step=0.1, help="")

    def step_4_interface():
        aug_dict = {
            False: "No",
            True: "Yes"
            }
        img_aug_list = ["random_horizontal_flip", "random_adjust_hue", "random_adjust_contrast", "random_adjust_saturation", "random_square_crop_by_scale", "random_scale_crop_and_pad_to_square"]
        with st.expander(label='Step 4: Image Dataset Augmentation (Optional)'):
            st.info("**Step 4:** Select image augmentation choice for the image dataset.")
            st.session_state["data_aug_status"] = st.selectbox(label="Choose whether or not to augment image dataset.", options=[False, True], format_func=lambda option: aug_dict[option], help="")
            if st.session_state["data_aug_status"]:
                st.session_state["data_augmentation"] = st.multiselect(label="Select one or more image augmentation choice.", options=img_aug_list, help="")
                random_square_crop_by_scale_interface = st.empty()
                random_scale_crop_and_pad_to_square_interface = st.empty()
                random_square_crop_by_scale = False
                random_scale_crop_and_pad_to_square = False
                for item in st.session_state["data_augmentation"]:
                    if item=="random_square_crop_by_scale":
                        random_square_crop_by_scale = True
                    if item=="random_scale_crop_and_pad_to_square":
                        random_scale_crop_and_pad_to_square = True
                if random_square_crop_by_scale:
                    with random_square_crop_by_scale_interface.container():
                        step_4_function_1()
                else: 
                    random_square_crop_by_scale_interface.empty()
                if random_scale_crop_and_pad_to_square:
                    with random_scale_crop_and_pad_to_square_interface.container():
                        step_4_function_2()
                else:
                    random_scale_crop_and_pad_to_square_interface.empty()

                st.markdown("""
                <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
                <p>
                Image augmentation will be implemented to image dataset during the training process.
                </p>
                </section>
                <p></p>
                """, unsafe_allow_html=True)  
            else:
                st.markdown("""
                <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
                <p>
                Image augmentation will not be implemented to image dataset during the training process.
                </p>
                </section>
                <p></p>
                """, unsafe_allow_html=True)    


    # 5. Change Optimizer and Learning Rate
    def step_5_interface():
        with st.expander(label='Step 5: Optimizer and Learning Rate (Optional)'):
            st.info('**Step 5:** Set the optimizer and learning rate of the model.')
            st.session_state["optimizer"] = st.selectbox(label="Select a type of optimizer", options=["momentum_optimizer", "adam_optimizer"],
                                                         format_func=lambda option: optimizer_dict[option], help="")
            if st.session_state["optimizer"]=="momentum_optimizer":
                if st.session_state["model"]==1:
                   default_warmup_lr = .013333
                   default_warmup_steps = 1000
                   default_learning_rate_base = 0.04
                   default_total_steps = st.session_state["num_steps"]

                elif st.session_state["model"]==2:
                   default_warmup_lr = 0.0
                   default_warmup_steps = 1000
                   default_learning_rate_base = 0.008
                   default_total_steps = st.session_state["num_steps"]

                elif st.session_state["model"]==3:
                   default_warmup_lr = 0.001
                   default_warmup_steps = 1000
                   default_learning_rate_base = 8e-2
                   default_total_steps = st.session_state["num_steps"]
            else:
                default_warmup_lr = 3e-4
                default_warmup_steps = 1000
                default_learning_rate_base = 1e-3
                default_total_steps = st.session_state["num_steps"]

            st.session_state["warmup_learning_rate"] = st.text_input(label="Enter a warmup learning rate value.", value=default_warmup_lr, help="")
            st.session_state["warmup_steps"] = st.slider(label="Select the number of warmup step.", min_value=0, max_value=st.session_state["num_steps"], value=default_warmup_steps, step=100, help="")
            st.session_state["learning_rate_base"] = st.text_input(label="Enter a learning rate base value.", value=default_learning_rate_base, help="")
            st.markdown("""
            <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
            <p>
            The chosen optimizer is <b>{}</b> and the learning rate are set to:<br>
            warmup learning rate is set to <b>{}</b><br>
            warmup steps is set to <b>{}</b><br>
            learning rate base is set to <b>{}</b><br>
            total steps is set to <b>{}</b>
            </p>
            </section>
            <p></p>
            """.format(st.session_state["optimizer"], st.session_state["warmup_learning_rate"], st.session_state["warmup_steps"], st.session_state["learning_rate_base"], default_total_steps), unsafe_allow_html=True)

    # 6. Change Dropout
    def step_6_interface():
        dp_dict = {
                    "false": "No",
                    "true": "Yes"
                    }
        with st.expander(label="Step 6: Set dropout (Optional)"):
            st.info('**Step 6:** Set the dropout keep probability of the model.')
            st.session_state["dropout"] = st.selectbox(label="Select whether to include dropout parameter.", options=["false", "true"],
                                                       format_func=lambda option: dp_dict[option] , help="Default dropout keep probability: 1, Reference: [EFF](https://eff.org)")
            if st.session_state["dropout"]=="true":
                st.session_state["dropout_keep_prob"] = st.slider(label="Select a dropout keep probability value", min_value=0.10, max_value=1.00, value=0.85, step=0.01)
                st.markdown("""
                <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
                <p>
                Dropout probability is set to <b>{}</b>.  
                </p>
                </section>
                <p></p>
                <p></p>
                """.format(round(1.00-st.session_state["dropout_keep_prob"], 2)), unsafe_allow_html=True)
            else:
                st.session_state["dropout_keep_prob"] = round(1.00)
                st.markdown("""
                <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
                <p>
                Dropout probability is not set.  
                </p>
                </section>
                <p></p>
                <p></p>
                """, unsafe_allow_html=True)

    # 7. Change batch size
    def step_7_interface():
         with st.expander(label="Step 7: Set training batch size"):
            st.info('**Step 7:** Set the batch size of dataset for each training steps.')
            batch_size = st.select_slider(label="Set batch size by dragging the slider left or right.",
                              options=[1, 2, 4, 8, 16, 32, 64], value=4,
                              help="Recommended value: 1")
            st.session_state["batch_size"] = batch_size
            st.markdown("""
            <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
            <p>
            The batch size is set to <b>{}</b>.  
            </p>
            </section>
            <p></p>
            <p></p>
            """.format(st.session_state["batch_size"]), unsafe_allow_html=True)

    # 8. Change max/min image size
    def step_8_interface():
         with st.expander(label="Step 8: Set image size"):
            st.info('**Step 8:** Set the minimum and maximum dimension of the uploaded image dataset.')
            st.session_state["tuple_img_dimension_config"] = st.slider(label="Set dimension range by dragging the slider left or right.",
                                                                       min_value=100, max_value=1024, value=(200,400))
            st.session_state["min_img_dimension_config"], st.session_state["max_img_dimension_config"] = st.session_state["tuple_img_dimension_config"][0], st.session_state["tuple_img_dimension_config"][1]
            st.markdown("""
            <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
            <p>
            Minimum dimension: <b>{}</b> &nbsp;&nbsp;&emsp;Maximum dimension: <b>{}</b>  
            </p>
            </section>
            <p></p>
            <p></p>
            """.format(st.session_state["min_img_dimension_config"], st.session_state["max_img_dimension_config"]), unsafe_allow_html=True)

    # 9. Choose or create a directory 
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

    def step_9_interface(main_path=rejoin_path):
        with st.expander(label="Step 9: Select a directory to save the model pipeline"):
            st.info('**Step 9:** Choose a directory to save the model configuration.')
            # Check if streamlit_user_directory exist
            streamlit_user_directory = main_path + '/streamlit_user_directory/'
            if not os.path.exists(streamlit_user_directory):
                os.mkdir(streamlit_user_directory)
            st.session_state["streamlit_user_dir_list"] = list_out_existing_user_directory(streamlit_user_directory)
            if not bool(st.session_state["streamlit_user_dir_list"]):
                st.markdown("""
                <section class='img_uploader_section' style='background-color:#ECD4FF; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
                <p>
                Oopsie, looks like no directory has been created yet.<br>
                Please create a directory and prepare required dataset by navigating to <b>Upload Data</b> page.
                </p>
                </section>
                <p></p>
                """.format(streamlit_user_directory + st.session_state["user_dir_choice"]), unsafe_allow_html=True)
            else:
                st.session_state["user_dir_choice"] =  st.selectbox(label="Please select or create a directory.",
                                                        options=st.session_state["streamlit_user_dir_list"])
                st.session_state["streamlit_user_final_dir_choice"] = streamlit_user_directory + st.session_state["user_dir_choice"] + "/"
                st.markdown("""
                <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
                <p>
                The selected directory is <b>{}</b>  
                </p>
                </section>
                <p></p>
                """.format(streamlit_user_directory + st.session_state["user_dir_choice"]), unsafe_allow_html=True)

    # Set image augmentation pipeline function
    def set_aug_param_false():
        configure_parameter.random_horizontal_flip = False
        configure_parameter.random_adjust_hue = False
        configure_parameter.random_adjust_contrast = False
        configure_parameter.random_adjust_saturation = False
        configure_parameter.random_square_crop_by_scale = False
        configure_parameter.random_scale_crop_and_pad_to_square = False

    def random_horizontal_flip():
        configure_parameter.random_horizontal_flip = True

    def random_adjust_hue():
        configure_parameter.random_adjust_hue = True

    def random_adjust_contrast():
        configure_parameter.random_adjust_contrast = True

    def random_adjust_saturation():
        configure_parameter.random_adjust_saturation = True


    def random_square_crop_by_scale():
        configure_parameter.random_square_crop_by_scale = True
        configure_parameter.min_scale_1 = st.session_state["min_scale_1"]
        configure_parameter.max_scale_1 = st.session_state["max_scale_1"]

    def random_scale_crop_and_pad_to_square():
        configure_parameter.random_scale_crop_and_pad_to_square = True
        configure_parameter.output_size = st.session_state["output_size"]
        configure_parameter.min_scale_2 = st.session_state["min_scale_2"]
        configure_parameter.max_scale_2 = st.session_state["max_scale_2"]

    aug_func_dict = {
        "random_horizontal_flip": random_horizontal_flip,
        "random_adjust_hue": random_adjust_hue,
        "random_adjust_contrast": random_adjust_contrast,
        "random_adjust_saturation": random_adjust_saturation,
        "random_square_crop_by_scale": random_square_crop_by_scale,
        "random_scale_crop_and_pad_to_square": random_scale_crop_and_pad_to_square
        }

    # Assembling all defined functions into a form
    config_pipeline_container = st.container()
    with config_pipeline_container:
        step_1_interface()
        step_2_interface()
        step_3_interface()
        step_4_interface()
        step_5_interface()
        step_6_interface()
        step_7_interface()
        step_8_interface()
        step_9_interface()
        submitted_configuration = st.button(label="Configure My Custom Model")
        if submitted_configuration:
            # 1. Set File Path
            streamlit_user_directory =  rejoin_path + '/streamlit_user_directory/' + st.session_state["user_dir_choice"] 
            model_count = 0
            for config_idx, config_file in enumerate(glob.glob(streamlit_user_directory + "/" + x_folder_subdir[3] +"/" + "*.config")):
                if config_idx>=0:
                    model_name = config_file.split("/")[-1]
                    model_type = model_name.split("_")[0]
                    if model_type==model_dict[st.session_state["model"]]:
                        model_count += 1
            model_count += 1
            st.session_state["streamlit_user_model_count"] = model_count
            configure_parameter.model = model_dict[st.session_state["model"]]
            configure_parameter.pipeline_fname = st.session_state["model_pipeline_path"]
            configure_parameter.label_map_pbtxt_fname = streamlit_user_directory + "/" +  x_folder_subdir[0] + "/" + st.session_state["user_dir_choice"] + "_label_map.pbtxt"
            configure_parameter.pipeline_writen_path = streamlit_user_directory + "/" + x_folder_subdir[3] + "/" + model_dict[st.session_state["model"]] + "_" + st.session_state["user_dir_choice"] + "_model_" + str( model_count) + ".config"

            # Check if the pretrained model has been downloaded
            pretrained_ckpt_fpath = rejoin_path + "/" + "streamlit_pretrained_models" + "/" + model_dict[st.session_state["model"]] + "/" + pretrained_model[model_dict[st.session_state["model"]]]
            pretrained_model_fpath = rejoin_path + "/" + "streamlit_pretrained_models" + "/" + model_dict[st.session_state["model"]]
            if not os.path.exists(rejoin_path + "/" + "streamlit_pretrained_models"):
                os.mkdir(rejoin_path + "/" + "streamlit_pretrained_models")
            if not os.path.exists(pretrained_model_fpath):
                os.mkdir(pretrained_model_fpath)
            if not os.path.exists(pretrained_ckpt_fpath + ".index"):
                st.markdown("""
                <section class="create_tf_record_warning_2" style="background-color:#ECD4FF; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px">
                <p>
                The <b>{}</b> pretrained model does not exists.<br>
                Please navigate to <b><i>User File Manager</i></b> page to download the pretrained model. 
                </p>
                </section>
                <p></p>""".format(model_dict[st.session_state["model"]]), unsafe_allow_html=True)
            else:
                if st.session_state["model"]==2:
                    key_1, key_2 = "_mask_train.record", "_mask_test.record"
                else:
                    key_1, key_2 = "_train.record", "_test.record"
                
                configure_parameter.fine_tune_checkpoint = pretrained_ckpt_fpath
                configure_parameter.train_record_fname = streamlit_user_directory + "/" +  x_folder_subdir[0] + "/" + st.session_state["user_dir_choice"] + key_1
                configure_parameter.test_record_fname = streamlit_user_directory + "/" +  x_folder_subdir[0] + "/" + st.session_state["user_dir_choice"] + key_2   
                streamlit_user_fine_tuned_path = streamlit_user_directory + "/" + x_folder_subdir[2] + "/" + model_dict[st.session_state["model"]] + "_" + st.session_state["user_dir_choice"] + "_model_" + str( model_count)
                streamlit_user_training_path = streamlit_user_directory + "/" + x_folder_subdir[1] + "/" + model_dict[st.session_state["model"]] + "_" + st.session_state["user_dir_choice"] + "_model_" + str( model_count)
                streamlit_user_node_red_path = streamlit_user_directory + "/" + x_folder_subdir[5] + "/" + model_dict[st.session_state["model"]] + "_" + st.session_state["user_dir_choice"] + "_model_" + str( model_count)


                if not os.path.exists(streamlit_user_fine_tuned_path):
                    os.mkdir(streamlit_user_fine_tuned_path)
                if not os.path.exists(streamlit_user_training_path):
                    os.mkdir(streamlit_user_training_path)
                if not os.path.exists(streamlit_user_node_red_path):
                    os.mkdir(streamlit_user_node_red_path)
                    os.mkdir(streamlit_user_node_red_path + "/" + "telegram_img")
                    os.mkdir(streamlit_user_node_red_path + "/" + "bbox_img")
                    os.mkdir(streamlit_user_node_red_path + "/" + "csv_prediction_log")
            
                # 2. Set number of steps
                configure_parameter.num_steps = st.session_state["num_steps"]

                # 3. Set IOU Threshold
                configure_parameter.flag_A = True
                configure_parameter.iou_thres = st.session_state["iou_thres"]

                # 4. Set Image Augmentation
                if st.session_state["data_aug_status"]==False:
                    configure_parameter.flag_B = False
                else: 
                    configure_parameter.flag_B = True
                    set_aug_param_false()
                    for img_aug_item in st.session_state["data_augmentation"]:
                        aug_func_dict[img_aug_item]()

                # 5. Set optimizer and learning rate
                if st.session_state["optimizer"] == "momentum_optimizer":
                    configure_parameter.flag_C = False
                elif st.session_state["optimizer"] == "adam_optimizer":
                    configure_parameter.flag_C = True
                    configure_parameter.optimizer = st.session_state["optimizer"]

                configure_parameter.warmup_lr = st.session_state["warmup_learning_rate"]
                configure_parameter.warmup_steps = st.session_state["warmup_steps"]
                configure_parameter.lr_base = st.session_state["learning_rate_base"]
                configure_parameter.num_steps = st.session_state["num_steps"]

                # 6. Set dropout keep probability
                if st.session_state["dropout"]=="true":
                    configure_parameter.flag_D = True
                    configure_parameter.dropout = st.session_state["dropout"]
                    configure_parameter.dropout_keep_prob = st.session_state["dropout_keep_prob"]
                else:
                    configure_parameter.flag_D = False
                    configure_parameter.dropout_keep_prob = st.session_state["dropout_keep_prob"]

                # 7. Set batch size
                configure_parameter.batch_size = st.session_state["batch_size"]

                # 8. Set min and max image size
                configure_parameter.min_img_dimension_config = st.session_state["min_img_dimension_config"]
                configure_parameter.max_img_dimension_config = st.session_state["max_img_dimension_config"]

                # Call the function
                configure_parameter.configure_pipeline()
                streamlit_user_pipeline_fpath = streamlit_user_directory + "/" + x_folder_subdir[3] + "/" + model_dict[st.session_state["model"]] + "_" + st.session_state["user_dir_choice"] + "_model_" + str(st.session_state["streamlit_user_model_count"]) + ".config"
                st.markdown("""
                <section class='img_uploader_section' style='background-color:#B5EAD7; padding: 15px 10px 1px 10px; border-radius: 5px 5px 5px 5px'>
                <p>
                The model pipeline has been generated for training process. The directory is at:<br>
                <b>{}</b>
                </p>
                </section>
                <p></p>
                """.format(streamlit_user_pipeline_fpath), unsafe_allow_html=True)


















































