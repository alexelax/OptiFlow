#https://github.com/ultralytics/ultralytics
from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model("../Resources/infer_data/traffic1.mp4",show=True,device="cpu")  # predict on an image
#results = model("0",show=True)  # webcam

