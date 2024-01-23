import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5_faces')

# Define the image path
image_path = "your_image_path"

# Load the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference on the image
model.conf=0.25
model.iou=0.45
results = model(image_rgb)
results.show()

# Flipping the image somehow returns higher accuracy values!
#results2 = model(cv2.flip(image_rgb, 1))
#results2.show()

# Optional: run face mesh on the detected faces
'''
from poseutils import Mesher
mesher = Mesher()
lm = mesher.process(image_rgb)
print(lm)
'''