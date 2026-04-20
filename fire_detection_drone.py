import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO


# ==============================
# Distance Estimator
# ==============================
class DistanceEstimator:
    def __init__(self, focal_length=3.6, ref_size=10):
        self.f = focal_length
        self.r = ref_size

    def compute(self, bbox_height):
        if bbox_height <= 0:
            return None
        return round((self.f * bbox_height) / self.r, 2)


# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")


model = load_model()
estimator = DistanceEstimator()


# ==============================
# UI
# ==============================
st.set_page_config(page_title="Fire Detection Drone", layout="wide")

st.title("🔥 Fire Detection Web App")
st.write("Real-time fire detection using YOLOv8")

option = st.sidebar.selectbox("Choose Input", ["Upload Image", "Camera"])


# ==============================
# Upload Image Mode
# ==============================
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                height = y2 - y1
                distance = estimator.compute(height)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                label = f"Fire {conf:.2f}"
                if distance:
                    label += f" | {distance}m"

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        st.image(frame, channels="BGR")


# ==============================
# Camera Mode
# ==============================
elif option == "Camera":
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                height = y2 - y1
                distance = estimator.compute(height)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                label = f"Fire {conf:.2f}"
                if distance:
                    label += f" | {distance}m"

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()