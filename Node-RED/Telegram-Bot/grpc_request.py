import grpc
import time
import json
import PIL.Image
import numpy as np
from tensorflow_serving.apis import model_service_pb2_grpc, model_management_pb2, get_model_status_pb2, predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.config import model_server_config_pb2
from tensorflow import make_tensor_proto
from tensorflow.core.framework import types_pb2

# Input: 1. image_np 
#        2. grpc port 
#        3. model name
#
# Output: detection_boxes, detection_classes, detection_scores

class grpc_request():
    def __init__(self, image_np, grpc_port, model_name):
        self.image_np = image_np
        self.grpc_port = grpc_port
        self.model_name = model_name

    def config(self):
        serving_config = {
            "hostport": "localhost:{}".format(self.grpc_port),
            "max_message_length": 2000 * 1024 * 1024,
            "timeout": 300,
            "signature_name": "serving_default",
            "model_name": self.model_name,
            "shape": self.image_np.shape
        }
        return serving_config

    def main(self):

        serving_config = self.config()
        channel = grpc.insecure_channel(serving_config['hostport'],
                                        options=[
                ('grpc.max_send_message_length', serving_config['max_message_length']),
                ('grpc.max_receive_message_length', serving_config['max_message_length'])
                ])
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = serving_config['model_name']
        request.model_spec.signature_name = serving_config['signature_name']
        request.inputs['input_tensor'].CopyFrom(make_tensor_proto(self.image_np, shape=[1] + list(serving_config['shape'])))
        start_time = time.time()
        response = stub.Predict(request, serving_config['timeout'])
        end_time = time.time()
        channel.close()
        total_time = end_time-start_time

        # Extract results
        detection_boxes = response.outputs['detection_boxes'].float_val
        detection_classes = response.outputs['detection_classes'].float_val
        detection_scores = response.outputs['detection_scores'].float_val

        return total_time, detection_boxes, detection_classes, detection_scores








