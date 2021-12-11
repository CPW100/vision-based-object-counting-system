import regex as re
import streamlit as st
from object_detection.utils import label_map_util

class Pipeline_Config():
    def __init__(self):

        self.flag_A = False  # Change IOU threshold in {batch_non_max_suppression}
        self.flag_B = False  # Image Augmentation
        self.flag_C = False  # change to Adam Optimizer & learning rate
        self.flag_D = False  # change dropout
        self.model = "Faster-RCNN-ResNet-50" 
        # model choice: ['Faster-RCNN-ResNet-50', 'Mask-RCNN', 'SSD-EfficientDet']
        self.label_map_pbtxt_fname = "Please enter your file path." 
        self.pipeline_fname = "Please enter your file path."
        self.pipeline_writen_path = "Please enter your file path."
        self.fine_tune_checkpoint = "Please enter your file path."
        self.train_record_fname = "Please enter your file path."
        self.test_record_fname = "Please enter your file path."
        self.batch_size = 1
        self.num_steps = 5000
        self.warmup_steps = 2000
        self.min_img_dimension_config = 480
        self.max_img_dimension_config = 1042
        self.dropout = 'true'
        self.dropout_keep_prob = 0.85
        self.pad_to_max_dimension = 'true'
        self.optimizer = 'adam_optimizer'
        self.epsilon = 1e-7
        self.lr_base = 1e-3
        self.warmup_lr = 3e-4
        self.data_aug = 'autoaugment_image' 
        self.aug_version = "v1"
        self.min_scale_1 = 0.6
        self.max_scale_1 = 1.3
        self.min_scale_2 = 0.1
        self.max_scale_2 = 2.0
        self.iou_thres = 0.8
        self.output_size = 512
        self.random_horizontal_flip = True
        self.random_adjust_hue = True
        self.random_adjust_contrast = True
        self.random_adjust_saturation = True
        self.random_square_crop_by_scale = True
        self.random_scale_crop_and_pad_to_square = True

    def get_num_classes(self):
        
        label_map = label_map_util.load_labelmap(self.label_map_pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return len(category_index.keys())

    def configure_pipeline(self):

        print('writing custom configuration file')
        num_classes = self.get_num_classes()
        with open(self.pipeline_fname) as f:
            s = f.read()
            pipeline_writen_path = self.pipeline_writen_path
        with open(pipeline_writen_path, 'w') as f:

            # # Auto augmentation (more suitable for coco-dataset)
            # s = re.sub('data_augmentation_options {\n    [a-z]+\_[a-z]+\_[a-z]+\s+\{',
            #            'data_augmentation_options {\n    ' + '{}'.format(data_aug) + '{\n      ' + '{}: "{}"'.format('policy_name', aug_version),s)

            # Change IOU threshold in {batch_non_max_suppression}
            if self.flag_A == True: 
                 s = re.sub('\siou_threshold: [0-9]+\.[0-9]+',
                            ' iou_threshold: {}'.format(self.iou_thres), s)
            else:
                pass

            # Image Augmentation
            if self.flag_B == True:
                if self.random_horizontal_flip:
                    aug_1 = 'data_augmentation_options {\n    random_horizontal_flip {\n    }\n  }'
                else:
                    aug_1 = None
                if self.random_adjust_hue:
                    aug_2 = 'data_augmentation_options {\n    random_adjust_hue {\n    }\n  }'
                else:
                    aug_2 = None
                if self.random_adjust_contrast:
                    aug_3 = 'data_augmentation_options {\n    random_adjust_contrast {\n    }\n  }'
                else:
                    aug_3 = None
                if self.random_adjust_saturation:
                    aug_4 = 'data_augmentation_options {\n    random_adjust_saturation {\n    }\n  }'
                else:
                    aug_4 = None
                if  self.random_square_crop_by_scale:
                    aug_5 = 'data_augmentation_options {\n    random_square_crop_by_scale {\n      ' + 'scale_min: {}\n      '.format(self.min_scale_1) + 'scale_max: {}\n    '.format(self.max_scale_1) + '}\n  }'
                else:
                    aug_5 = None
                if self.random_scale_crop_and_pad_to_square:
                    aug_6 = 'data_augmentation_options {\n    random_scale_crop_and_pad_to_square {\n      ' + 'output_size: {}\n      '.format(self.output_size) + 'scale_min: {}\n      '.format(self.min_scale_2) + 'scale_max: {}\n    '.format(self.max_scale_2) + '}\n  }'
                else:
                    aug_6 = None

                if self.model=='SSD-EfficientDet':
                    aug_str = [aug_1, aug_2, aug_3, aug_4, aug_5]
                else:
                    aug_str = [aug_1, aug_2, aug_3, aug_4, aug_5, aug_6]

                data_aug_str = ""
                item_count = 0
                icount = 0
                for item in aug_str:
                    if item is not None:
                        item_count += 1
            
                if item_count==1:
                    for item in aug_str:
                      if item is not None:
                        data_aug_str = item

                elif item_count>1:
                    for item in (aug_str):
                        if item is not None:
                            icount += 1
                            if icount==item_count:
                                data_aug_str += item
                            else:
                                data_aug_str += item + '\n\n  '
                if self.model=='SSD-EfficientDet':
                    if aug_6==None:
                        item = ""
                    else:
                        item=aug_6
                    s = re.sub('data_augmentation_options {\n    random_scale_crop_and_pad_to_square {\n      ' + 
                               'output_size: [\d]+\n      ' + 
                               'scale_min: [\d\.]+\n      ' + 
                               'scale_max: [\d\.]+\n    ' + '}\n  }', item, s)

                    s = re.sub('data_augmentation_options {\n    random_horizontal_flip {\n    }\n  }', data_aug_str, s)

                else:
                    s = re.sub('data_augmentation_options {\n    random_horizontal_flip {\n    }\n  }', data_aug_str, s)

            else:
                if self.model=='SSD-EfficientDet':
                   s = re.sub('data_augmentation_options {\n    random_scale_crop_and_pad_to_square {\n      ' + 
                               'output_size: [\d]+\n      ' + 
                               'scale_min: [\d\.]+\n      ' + 
                               'scale_max: [\d\.]+\n    ' + '}\n  }', "", s)
                   s = re.sub('data_augmentation_options {\n    random_horizontal_flip {\n    }\n  }', "", s)
                else:
                   s = re.sub('data_augmentation_options {\n    random_horizontal_flip {\n    }\n  }', "", s)

            # change to Adam Optimizer & learning rate
            # """ https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/configs/tf2/center_net_deepmac_1024x1024_voc_only_tpu-128.config"""
            if self.flag_C == True:
                 s = re.sub('optimizer {\n    [a-z]+\_[a-z]+\:\s+\{',
                            'optimizer {\n    ' + '{}:'.format(self.optimizer) + '{' + '\n      {}: {}'.format('epsilon', float(1e-7)),s)
                 s = re.sub('learning_rate_base: \.[\w\-+]+',
                            'learning_rate_base: {}'.format(self.lr_base), s)
                 s = re.sub('warmup_learning_rate: \.[0-9]+',
                            'warmup_learning_rate: {}'.format(self.warmup_lr), s)
                 s = re.sub('warmup_steps: [0-9]+\n        }\n      }\n      [a-z]+\_[a-z]+\_[a-z]+\:\s+[0-9]+\.[0-9]+\n\s+\s+\s+\s+\}',
                            'warmup_steps: 2000\n        }\n      }\n    ' + '}',s)
                 s = re.sub('total_steps: [0-9]+',
                            'total_steps: {}'.format(self.num_steps),s)
                 s = re.sub('warmup_steps: [0-9]+',
                            'warmup_steps: {}'.format(self.warmup_steps),s)
            else:
                 s = re.sub('learning_rate_base: \.[\w\-+]+',
                            'learning_rate_base: {}'.format(self.lr_base), s)
                 s = re.sub('warmup_learning_rate: \.[0-9]+',
                            'warmup_learning_rate: {}'.format(self.warmup_lr), s)
                 s = re.sub('total_steps: [0-9]+',
                            'total_steps: {}'.format(self.num_steps),s)
                 s = re.sub('warmup_steps: [0-9]+',
                            'warmup_steps: {}'.format(self.warmup_steps),s)

            # change dropout
            if self.flag_D == True:
                if self.model == "SSD-EfficientDet":
                     s = re.sub('weight_shared_convolutional_box_predictor {', 
                                'weight_shared_convolutional_box_predictor {\n        ' + 
                                'use_dropout: {}'.format(self.dropout) + '\n        ' +
                                'dropout_keep_probability: {}'.format(self.dropout_keep_prob),s)

                else:
                     s = re.sub('use_dropout: false',
                               'use_dropout: {}'.format(self.dropout), s)
                     s = re.sub('dropout_keep_probability: [+-]?[0-9]+\.[0-9]+',
                                'dropout_keep_probability: {}'.format(self.dropout_keep_prob),s)
            else:
                pass

            # resize image
            if self.model == "Mask-RCNN":
                s = re.sub('fixed_shape_resizer {',
                           '{}'.format('keep_aspect_ratio_resizer {'), s)
                s = re.sub('[^_]height: [0-9]+',
                           'min_dimension: {}'.format(self.min_img_dimension_config), s)
                s = re.sub('[^_]width: [0-9]+',
                           'max_dimension: {}'.format(self.max_img_dimension_config) + '\n       ' + 
                           'pad_to_max_dimension: {}'.format(self.pad_to_max_dimension), s)

            elif self.model=='SSD-EfficientDet':
                pass

            else:
                s = re.sub('min_dimension: [0-9]+',
                            'min_dimension: {}'.format(self.min_img_dimension_config), s)
                s = re.sub('max_dimension: [0-9]+',
                            'max_dimension: {}'.format(self.max_img_dimension_config), s)
   

            # fine_tune_checkpoint
            s = re.sub('fine_tune_checkpoint: [\w\/.""-]+',
                       'fine_tune_checkpoint: "{}"'.format(self.fine_tune_checkpoint), s)
    
            # tfrecord files train and test.
            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(self.train_record_fname), s)
            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(self.test_record_fname), s)

            # label_map_path
            s = re.sub(
                'label_map_path: ".*?"', 'label_map_path: "{}"'.format(self.label_map_pbtxt_fname), s)

            # Set training batch_size.
            s = re.sub('batch_size: [\d\;]+',
                       'batch_size: {}'.format(self.batch_size), s)

            # Set training steps, num_steps
            s = re.sub('num_steps: [0-9]+',
                       'num_steps: {}'.format(self.num_steps), s)
    
            # Set number of classes num_classes.
            s = re.sub('num_classes: [0-9]+',
                       'num_classes: {}'.format(num_classes), s)
    
            #fine-tune checkpoint type
            s = re.sub(
                'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
        
            f.write(s)