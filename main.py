import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained gender classification model
gender_model = tf.keras.models.load_model('gender_classification_model.h5')

# Function to select an image file using Streamlit file uploader
def select_image_file():
    st.sidebar.title('Select an Image')
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
    )
    if uploaded_file is not None:
        # Convert the file to an OpenCV image.
        image = Image.open(uploaded_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    return None

# Load YOLO model
model = YOLO("yolov8l.pt")

# Main Streamlit app code
def main():
    st.title('YOLO Object Detection with Gender Classification')

    # Select an image
    img = select_image_file()

    if img is not None:
        # Resize image to 1280x720
        img_resized = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

        # Perform object detection with YOLO
        results = model(img_resized)

        # Get the annotated image from results
        annotated_img = results[0].plot()

        # Convert BGR image to RGB for displaying in Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Perform gender classification on detected faces
        for (x, y, w, h) in faces:
            face = img_resized[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_normalized = face_resized / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            gender_prediction = gender_model.predict(face_expanded)
            gender_label = 'Male' if gender_prediction[0][0] > 0.5 else 'Female'
            color = (0, 255, 0) if gender_label == 'Male' else (255, 0, 0)
            cv2.rectangle(annotated_img_rgb, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated_img_rgb, gender_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display original and annotated images
        st.image([img_rgb, annotated_img_rgb], caption=['Original Image', 'Annotated Image'], width=640)

# Run the Streamlit app
if __name__ == '__main__':
    main()
