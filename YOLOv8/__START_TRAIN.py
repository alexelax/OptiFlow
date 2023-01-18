from ultralytics import YOLO

# Load a model
model = YOLO("_train/yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="_train/custom.yaml", epochs=3)  # train the model
success = model.export(format="onnx")  # export the model to ONNX format