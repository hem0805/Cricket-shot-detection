import streamlit as st
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
from moviepy.editor import VideoFileClip
import tempfile
import cv2
from PIL import Image
import os

plot_path = 'training_plots.png'
if os.path.exists(plot_path):
    st.sidebar.header("Model Training Progress")
    st.sidebar.image(plot_path, use_column_width=True)


# Load trained model and label encoder
model = tf.keras.models.load_model('cricket_shot_lstm_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
sequence_length = 30

def process_frame(frame, sequence_buffer):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    pred_label = None
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        for lm in landmarks:
            keypoints.extend([lm.x, lm.y, lm.z])
        sequence_buffer.append(keypoints)
        if len(sequence_buffer) == sequence_length:
            input_data = np.expand_dims(sequence_buffer, axis=0)
            predictions = model.predict(input_data, verbose=0)
            pred_class = np.argmax(predictions, axis=1)[0]
            pred_label = le.inverse_transform([pred_class])[0]
            # cv2.putText(frame, f'Shot: {pred_label}', (30, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    return frame, pred_label

st.title("Cricket Shot Detection with Streamlit")

uploaded_file = st.file_uploader("Upload a cricket video or image", type=["mp4","avi","mov","mkv","jpg","jpeg","png"])

shot_result = None  # Store display string here

if uploaded_file:
    if uploaded_file.type.startswith('video'):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        temp_file.close()
        sequence_buffer = deque(maxlen=sequence_length)
        clip = VideoFileClip(temp_file.name)
        stframe = st.empty()
        last_pred = None

        for frame in clip.iter_frames():
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_frame, pred = process_frame(frame_bgr, sequence_buffer)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            stframe.image(output_frame)
            if pred:
                last_pred = pred  # Update last prediction

        # Show the final detected shot below video
        if last_pred:
            shot_result = f"Shot played: {last_pred}"
        else:
            shot_result = "No shot detected."
    else:
        img = np.array(Image.open(uploaded_file).convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        sequence_buffer = deque(maxlen=sequence_length)
        output_frame, pred = process_frame(img_bgr, sequence_buffer)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        st.image(output_frame)
        if pred:
            shot_result = f"Shot played: {pred}"
        else:
            shot_result = "No shot detected."

    # Display the result below
    st.markdown(f"**{shot_result}**")



