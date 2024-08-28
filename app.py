# Import necessary libraries
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile

# Load the pre-trained model
model = load_model('./hand_gesture_recognition_model.h5')

# Define the mapping of class indices to gesture names
gesture_names = {
    0: 'Index Pointing Up',
    1: 'Palm Down',
    2: 'Fist',
    3: 'Thumbs Down',
    4: 'Thumbs Up',
    5: 'Palm Up',
    6: 'Victory',
    7: 'Stop',
    8: 'OK',
    9: 'Call Me'
}

# Function to preprocess a frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize frame to match model input
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension (64, 64, 1)
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 64, 64, 1)
    return frame

# Streamlit UI to upload a video file
st.title("Hand Gesture Recognition from Uploaded Video")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Initialize video capture for the uploaded file
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Create a placeholder for video frames

    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            break
        
        processed_frame = preprocess_frame(frame)  # Preprocess the frame
        
        # Make prediction using the model
        prediction = model.predict(processed_frame)
        predicted_class = np.argmax(prediction, axis=1)[0]
        gesture = gesture_names[predicted_class]  # Map class index to gesture name
        
        # Display the gesture on the video
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
        # Display the frame using Streamlit
        stframe.image(frame, channels="BGR")
    
    cap.release()  # Release video capture
else:
    st.write("Please upload a video file.")
