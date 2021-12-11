import io
import json
import xmltodict
import streamlit as st
import pandas as pd
import tensorflow as tf
from collections import namedtuple
from PIL import Image
from io import BytesIO
from object_detection.utils import dataset_util, label_map_util


def xml_to_csv(uploaded_files, option=None):
    if uploaded_files:
        xml_list = []
        for xml_file in uploaded_files:
            xml = xml_file.read()
            json_data = json.loads(json.dumps(xmltodict.parse(xml)))
            if isinstance(json_data['annotation']['object'], dict):
                value = (str(json_data['annotation']['filename']),
                         int(json_data['annotation']['size']['width']),
                         int(json_data['annotation']['size']['height']),
                         str(json_data['annotation']['object']['name']),
                         int(json_data['annotation']['object']['bndbox']['xmin']),
                         int(json_data['annotation']['object']['bndbox']['ymin']),
                         int(json_data['annotation']['object']['bndbox']['xmax']),
                         int(json_data['annotation']['object']['bndbox']['ymax']))
                xml_list.append(value)

            elif isinstance(json_data['annotation']['object'], list): 
                for element in json_data['annotation']['object']:  
                    value = (str(json_data['annotation']['filename']),
                             int(json_data['annotation']['size']['width']),
                             int(json_data['annotation']['size']['height']),
                             str(element['name']),
                             int(element['bndbox']['xmin']),
                             int(element['bndbox']['ymin']),
                             int(element['bndbox']['xmax']),
                             int(element['bndbox']['ymax']))
                    xml_list.append(value)

            column_name = ['filename', 'width', 'height',
                            'class', 'xmin', 'ymin', 'xmax', 'ymax']
            xml_df = pd.DataFrame(xml_list, columns=column_name)
            if option=="Train":
                st.session_state['xml_df_train'] = xml_df
                key = "XML_files_train"
            elif option=="Test":
                st.session_state['xml_df_test'] = xml_df
                key = "XML_files_test"

    else:
        if option=="Train":
            st.session_state['xml_df_train'] = None
        elif option=="Test":
            st.session_state['xml_df_test'] = None



def split(df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def class_text_to_int(label_map_dict, row_label):
     return label_map_dict[row_label]

def create_tf_example(xml_df, imgs, imgs_name, encoded_jpg, csv_output_path, tf_output_path, label_map_path):
    if xml_df is not None and imgs is not None:
        label_map = label_map_util.load_labelmap(label_map_path)
        label_map_dict = label_map_util.get_label_map_dict(label_map)
        data = namedtuple('data', ['filename', 'object'])
        gb = xml_df.groupby('filename')
        grouped = split(xml_df, 'filename')
        writer = tf.io.TFRecordWriter(tf_output_path)
        for group in grouped:
            for i, img_file in enumerate(imgs):
                if group.filename==imgs_name[i]:
                    width, height = img_file.size

                    # Initialize some variables
                    filename = group.filename.encode('utf8')
                    image_format = b'jpg'
                    xmins = []
                    xmaxs = []
                    ymins = []
                    ymaxs = []
                    classes_text = []
                    classes = []

                    for index, row in group.object.iterrows():
                        xmins.append(row['xmin'] / width)
                        xmaxs.append(row['xmax'] / width)
                        ymins.append(row['ymin'] / height)
                        ymaxs.append(row['ymax'] / height)
                        classes_text.append(row['class'].encode('utf8'))
                        classes.append(class_text_to_int(label_map_dict, row['class']))

                    tf_example = tf.train.Example(features=tf.train.Features(feature={
                        'image/height': dataset_util.int64_feature(height),
                        'image/width': dataset_util.int64_feature(width),
                        'image/filename': dataset_util.bytes_feature(filename),
                        'image/source_id': dataset_util.bytes_feature(filename),
                        'image/encoded': dataset_util.bytes_feature(encoded_jpg[i]), 
                        'image/format': dataset_util.bytes_feature(image_format),
                        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                        'image/object/class/label': dataset_util.int64_list_feature(classes),
                    }))
                    writer.write(tf_example.SerializeToString())
        writer.close()
        if csv_output_path is not None:
            xml_df.to_csv(csv_output_path, index=None)
            print('Dataset CSV Status: Successfully created the CSV file: {}'.format(csv_output_path))
    else:
        print("Dataset TFRecords Status: Please complete Step 2, 3 and 4 before proceeding with Step 5.")