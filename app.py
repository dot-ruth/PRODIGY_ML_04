import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model = load_model('./hand_gesture_recognition_model.h5')  # Update with your model path

# Define the image size
img_size = (64, 64)

# Define the mapping of class indices to labels
class_labels = {
    0: 'Palm Up',
    1: 'Fist',
    2: 'Thumb Up',
    3: 'Open Hand',
    4: 'Palm Down',
    5: 'Peace Sign',
    6: 'Pointing Up',
    7: 'Pointing Down',
    8: 'Thumb Down',
    9: 'OK Sign',
    # Add more mappings as needed
}

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, img_size)  # Resize the image
    image = img_to_array(image)  # Convert image to numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

# Streamlit app
st.title('Video-Based Hand Gesture Recognition')

# Use webcam or video file
video_source = st.selectbox("Choose video source", ("Webcam", "Upload Video"))

if video_source == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        video = uploaded_file.name
else:
    video = 0  # 0 means webcam

if st.button('Start Detection'):
    stframe = st.empty()
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_image(frame)

        # Make prediction
        predictions = model.predict(processed_frame)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels.get(predicted_class_index, "Unknown Gesture")

        # Display the result on the frame
        cv2.putText(frame, f"Gesture: {predicted_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        stframe.image(frame, channels="BGR")

    cap.release()
