import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Detection", layout="centered")

st.title("🔍 Face Detection")
st.subheader("Use webcam or upload an image to detect faces.")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ----------------------- Function: Face Detection in Uploaded Image -----------------------
def detect_faces_in_image(uploaded_image):
    # Read and convert the image
    img = Image.open(uploaded_image)
    img_array = np.array(img.convert('RGB'))
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    result_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(result_img, channels="RGB", caption="Detected Faces")


# ----------------------- Function: Face Detection via Webcam -----------------------
def detect_faces_with_webcam():
    stframe = st.empty()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Unable to access the webcam.")
        return

    stop_button = st.button("Stop Camera")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠ Could not read from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        stframe.image(frame, channels="RGB")

        if stop_button:
            break

    cap.release()


# ----------------------- Streamlit UI -----------------------

# Webcam button
if st.button("📷 Open Webcam and Detect Faces"):
    detect_faces_with_webcam()

# Image upload
uploaded_image = st.file_uploader("📁 Or upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    detect_faces_in_image(uploaded_image)
