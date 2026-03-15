import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model_loader import load_model
from detector import run_detection
import tempfile
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="YOLOv8 Detection App")

st.title("YOLOv8 Object Detection")

# Load YOLO model
model = load_model("yolov8n")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Image Detection", "Video Detection", "Realtime Webcam"]
)

conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)

# ---------------- IMAGE DETECTION ----------------

if mode == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        frame = np.array(image)

        frame = run_detection(frame, model, conf)

        st.image(frame, caption="Detected Image")

# ---------------- VIDEO DETECTION ----------------

elif mode == "Video Detection":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame = run_detection(frame, model, conf)

            stframe.image(frame, channels="BGR")

        cap.release()

# ---------------- REALTIME WEBCAM ----------------

elif mode == "Realtime Webcam":

    st.header("Realtime Webcam Detection")

    def video_frame_callback(frame):

        img = frame.to_ndarray(format="bgr24")

        img = run_detection(img, model, conf)

        return img

    webrtc_streamer(
        key="webcam",
        video_frame_callback=video_frame_callback
    )