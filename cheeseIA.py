import cv2
import numpy as np
# import tensorflow as tf
import keras
import os

from imageai.Detection.Custom import CustomObjectDetection

execution_path = os.getcwd()

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "data/camembert/models/detection_model-ex-001--loss-0045.783.h5"))
detector.setJsonPath(os.path.join(execution_path, "data/camembert/json/detection_config.json"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "images/téléchargement.jpeg"),
                                             output_image_path=os.path.join(execution_path, "results/camembert2_result.jpeg"))

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])

# cap = cv2.VideoCapture(0)

# if cap.isOpened():
#  global_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#  global_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
