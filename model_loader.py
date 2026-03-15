from ultralytics import YOLO

def load_model(model_path="yolov8n"):
    """
    Load YOLOv8 model.
    Default: yolov8n.pt (nano, small and fast for webcam)
    """
    model = YOLO(model_path)
    return model