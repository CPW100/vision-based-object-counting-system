r"""
A module to run object detection model and 
send it to Node-RED.
Reference:
https://pypi.org/project/paho-mqtt/
"""
import time
import io
import os
import paho.mqtt.client as mqtt

from PIL import Image
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime('%d/%m/%Y %H:%M:%S')

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("test")
    client.subscribe("img")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print("   " + msg.topic+" "+str(msg.payload))

# The callback when the client disconnect from the broker
def on_disconnect(client, userdata, rc):
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Disconnected from broker.")

# The callback when the client publish a message
def on_publish(client, userdata, mid):
    print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Message is published.")
    print(f"    client: {client}")
    print(f"    userdata: {userdata}")
    print(f"    mid: {mid}")


broker_ip = "localhost" # local host
broker_port = 1883
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.on_publish = on_publish

# Read and convert image into byte array
my_img = Image.open('D:/Program_Files/VisualStudio2019/streamlit_project/streamlit_user_directory/apple/node-red/Faster-RCNN-ResNet-50_apple_model_1/bbox_img/img_1_bb.jpg')
buf = io.BytesIO()
my_img.save(buf, format='JPEG')
byte_im = buf.getvalue()

print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Connecting to MQTT broker")
client.connect(broker_ip, broker_port, 60)
print(f"{now.strftime('%d/%m/%Y %H:%M:%S')}: Connected to MQTT broker")
client.loop_start()
client.publish(topic='test', payload="Hello, I am a Python client publisher.")
client.publish(topic='img', payload=byte_im)
time.sleep(4)
client.loop_stop()
client.disconnect()

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
# client.loop_forever()
