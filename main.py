import cv2
from model_loader import load_model
from detector import run_detection

def main():
    # Load YOLOv8 model
    model = load_model("yolov8n.pt")  # change to your trained model if needed

    # Open webcam
    cap = cv2.VideoCapture(0)  # default camera index
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLO detection
        frame = run_detection(frame, model, conf_thresh=0.3)

        # Display
        cv2.imshow("YOLOv8 Webcam Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()