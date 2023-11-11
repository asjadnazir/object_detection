from ultralytics import YOLO
import torch


def train():
    print('Training Start')
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolov8n.yaml")
    # model.to(device)
    results = model.train(data="config.yaml", epochs=10)
    print('Training End')
