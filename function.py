# Ensure you have the appropriate import for your YOLO implementation
from ultralytics import YOLO
import requests
from io import BytesIO
from PIL import Image
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


class DetectObject:
    def __init__(self, model_path):
        model_path = 'best.pt'
        self.model = YOLO(model_path)
        self.IMG_SIZE = 640

    def detect_image(self, url, threshold):
        CONF_THRESHOLD = threshold

        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        # convert image to np because yolo only support np.array
        img = np.array(img)
        results = self.model.predict(
            source=img, imgsz=self.IMG_SIZE, conf=CONF_THRESHOLD)
        for result in results:
            for detection in result.boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                label = detection.cls[0]  # Class label
                conf = detection.conf[0]  # Confidence score
                # print(f'Label is {label} with confidence score {conf:.2f}')
                return conf

    def display_image(self, url, threshold):
        CONF_THRESHOLD = threshold

        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = np.array(img)
        results = self.model.predict(
            source=img, imgsz=self.IMG_SIZE, conf=CONF_THRESHOLD)
        for result in results:
            len_box = len(result.boxes)
            # detect.append(len_box)
        for detection in result.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            label = detection.cls[0]  # Class label
            conf = detection.conf[0]  # Confidence score
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Draw label and confidence score
            cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Display the image inline
        plt.imshow(img_rgb)
        plt.axis('off')  # Hide axes
        plt.show()
