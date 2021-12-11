import os
import sys
import pandas as pd
import altair as alt
import urllib.request
import subprocess
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

def app():
    # Specify main directory
    main_dir = os.getcwd()
    split_path = main_dir.split("/")
    split_path = split_path[:len(split_path)-1]
    separator = "/"
    rejoin_path = separator.join(split_path)
    streamlit_user_directory = rejoin_path + "/" + "streamlit_user_directory"
    streamlit_eval_directory = rejoin_path + "/" + "streamlit_result_directory"
    x_folder_subdir = ("dataset", "training", "fine_tuned", "pipeline", "evaluation", "node-red")
    if not os.path.exists(streamlit_eval_directory):
        os.mkdir(streamlit_eval_directory)
    # Streamlit Interface
    st.subheader("""Model Performance""")
    
    # Initializing Session State
    session_state_keys = {"chosen_streamlit_directory": None,}
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

    def visualize_model_performance_interface():
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
            streamlit_user_eval_directory = streamlit_user_directory + "/" + st.session_state["chosen_streamlit_directory"] + "/" + x_folder_subdir[4]
            
            # Create Panda Dataframe to record evaluated result
            all_eval_pd = pd.DataFrame(columns = ['Dataset', 'Model', 'Accuracy', 
                                                  'False Positive Detection',
                                                  'Total time taken (s)',
                                                  'Time per detection (s)'])
            short_name_list = []
            for dataset_dir in streamlit_user_dir_list:
                eval_model_dir_list = list_out_existing_user_directory(streamlit_user_directory + "/" + dataset_dir +
                                                                       "/" + "evaluation")
                for model_dir in eval_model_dir_list:
                    user_eval_dir = streamlit_user_directory + "/" + dataset_dir + "/evaluation/" + model_dir
                    eval_csv_fpath = user_eval_dir + "/evaluated_csv_result/average_eval_result.csv"
                    if os.path.exists(eval_csv_fpath):
                        short_name_list.append(model_dir.split('-')[0] + "_" + model_dir.split('_')[-1])
                        eval_df = pd.read_csv(eval_csv_fpath)
                        temp_dict = {'Dataset': dataset_dir, 
                                     'Model': model_dir, 
                                     'Model Short Name': model_dir.split('-')[0] + "_" + model_dir.split('_')[-1],
                                     'Accuracy': eval_df.iloc[0]['Accuracy'], 
                                     'False Positive Detection': eval_df.iloc[0]['False Positive Detection'],
                                     'Total time taken (s)': eval_df.iloc[0]['Total time taken (s)'],
                                     'Time per detection (s)': eval_df.iloc[0]['Time per detection (s)']}
                        all_eval_pd = all_eval_pd.append(temp_dict, ignore_index=True)
            all_eval_csv_path = streamlit_eval_directory + "/" + "model_performance.csv"        
            all_eval_pd.to_csv(all_eval_csv_path, index=False, index_label=False)
            
            grouped = all_eval_pd.groupby('Dataset')
            for df_name, df_group in grouped:
                if df_name==st.session_state["chosen_streamlit_directory"]:
                    st.write(df_group)
                    my_df = df_group
                    my_df.loc[:, 'Accuracy'] *=100
                    my_df.loc[:, 'False Positive Detection'] *=100
                    my_df.loc[:, 'Time per detection (s)'] *=1000
                    break
                
            my_df['Time per detection (s)'] = my_df['Time per detection (s)'].round(decimals=2)
            
            # Plot 1 - Accuracy
            width = st.sidebar.slider("plot width", 0.1, 25., 3.)
            height = st.sidebar.slider("plot height", 0.1, 25., 1.)
            fontSize = st.sidebar.slider("axis font size", 1, 25, 5)
            labelfontSize = st.sidebar.slider("label font size", 1, 25, 5)
            barwidth = st.sidebar.slider("bar size", 0.1, 5., 0.2)
            
            # Altair bar 1
            bar1 = alt.Chart(my_df, title="Average Accuracy of Models (%)").mark_bar(size=25, color='mediumorchid', opacity=0.9).encode(x=alt.X('Model Short Name', title="", axis=alt.Axis(labelAngle=-45)), y=alt.Y('Accuracy' , title='Average Accuracy (%)', axis=alt.Axis(values=np.arange(0,110,10)))).configure_axis(labelFontSize=25,titleFontSize=25).configure_title(fontSize=25).properties(width=1000,height=500)
            st.altair_chart(bar1, use_container_width=True)
    
            

            # Plot 2 - Detection Time
            bar2 = alt.Chart(my_df, title="Time per Object detection (ms)").mark_bar(color="turquoise", size=25).encode(
                x=alt.X('Time per detection (s):Q' , title='Detection Time (ms)', axis=alt.Axis(values=np.arange(0.,max(my_df['Time per detection (s)'] + 1),4.0))), 
                y=alt.Y('Model Short Name:O', title=""))
            
            text2 = bar2.mark_text(
                align='left',
                baseline='middle',
                dx=3,  # Nudges text to right so it doesn't appear on top of the bar
                size=18
            ).encode(
                text='Time per detection (s):O'
            )
            st.altair_chart((bar2 + text2).configure_title(fontSize=24).configure_axis(labelFontSize=25,titleFontSize=25).properties(width=1000,height=700), use_container_width=True)

            
            # Plot 3 - False Positive
            bar3 = alt.Chart(my_df, title="False Positive Rate (%)").mark_bar(size=25, color='mediumvioletred', opacity=0.9).encode(
                x=alt.X('Model Short Name:O', title="", axis=alt.Axis(labelAngle=-45)), 
                y=alt.Y('False Positive Detection:Q' , 
                        title='False Positive Rate (%)', 
                        axis=alt.Axis(values=np.arange(0.,round(max((100*my_df['False Positive Detection']).tolist())+0.2, 2),0.1))))
                        
            text3 = bar3.mark_text(align="center", baseline="bottom", dy=-3, size=18).encode(text=alt.Text("False Positive Detection:Q", format=",.2f"))
            st.altair_chart((bar3 + text3).configure_title(fontSize=24).configure_axis(labelFontSize=25,titleFontSize=25).properties(width=1000,height=700), use_container_width=True)
            
            
    visualize_model_performance_interface()
                    
                                                          


