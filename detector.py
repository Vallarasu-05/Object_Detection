import cv2

def run_detection(frame, model, conf_thresh=0.3):
    """
    Run YOLOv8 detection on a single frame and draw bounding boxes.
    Args:
        frame: image frame (numpy array)
        model: YOLOv8 model
        conf_thresh: confidence threshold to display boxes
    Returns:
        frame with bounding boxes drawn
    """
    results = model(frame)[0]

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = f"{model.names[cls_id]} {conf:.2f}"

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame