from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model

# Train the model
model.train(
    data='dataset.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    name='road_accident_detection'
)