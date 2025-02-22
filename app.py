import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np
import av

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (joint)
    c = np.array(c)  # Last point

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid numerical errors
    return np.degrees(angle)

# Define WebRTC video processor
class PoseEstimationProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get joint coordinates
            h, w, _ = img.shape
            def get_coords(landmark):
                return (int(landmark.x * w), int(landmark.y * h))

            joints = {
                "left_shoulder": get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]),
                "left_elbow": get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]),
                "left_wrist": get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST]),
                "left_hip": get_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP]),
                "right_shoulder": get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                "right_elbow": get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]),
                "right_wrist": get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]),
                "right_hip": get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
            }

            # Calculate angles
            angles = {
                "right_elbow": calculate_angle(joints["left_shoulder"], joints["left_elbow"], joints["left_wrist"]),
                "right_shoulder": calculate_angle(joints["left_hip"], joints["left_shoulder"], joints["left_elbow"]),
                "left_elbow": calculate_angle(joints["right_shoulder"], joints["right_elbow"], joints["right_wrist"]),
                "left_shoulder": calculate_angle(joints["right_hip"], joints["right_shoulder"], joints["right_elbow"])
            }

            # Draw angles on the frame
            for joint, angle in angles.items():
                position = joints[joint]
                cv2.putText(img, f"{int(angle)}°", position, cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 255), 2, cv2.LINE_AA)  # Yellow color

            # Draw landmarks
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Live Pose Estimation with Angle Calculation")
st.write("This app detects body posture using a webcam and calculates joint angles in real-time.")

# Start WebRTC video streaming
webrtc_streamer(key="pose-detection", video_processor_factory=PoseEstimationProcessor)
