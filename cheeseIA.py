import cv2
import numpy as np
# import tensorflow as tf
import keras
import os

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "images/fromages.jpg"),
                                             output_image_path=os.path.join(execution_path, "results/fromages_result.jpg"))

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])

# cap = cv2.VideoCapture(0)

# if cap.isOpened():
#  global_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#  global_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
