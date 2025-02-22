import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
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

# Streamlit UI
st.title("Web-Based Pose Estimation")
st.write("This app detects body posture using a webcam and calculates joint angles.")

# Create a Streamlit frame holder
frame_holder = st.empty()

# Access webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Failed to open webcam")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe Pose
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get required joint points for both left and right sides
            joints = {
                "left_shoulder": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                "left_elbow": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                "left_wrist": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                "left_hip": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                "right_shoulder": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                "right_elbow": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                "right_wrist": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                "right_hip": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            }

            # Calculate angles for both left and right arms
            angles = {
                "right_elbow": calculate_angle((joints["left_shoulder"].x, joints["left_shoulder"].y),
                                                (joints["left_elbow"].x, joints["left_elbow"].y),
                                                (joints["left_wrist"].x, joints["left_wrist"].y)),
                "right_shoulder": calculate_angle((joints["left_hip"].x, joints["left_hip"].y),
                                                   (joints["left_shoulder"].x, joints["left_shoulder"].y),
                                                   (joints["left_elbow"].x, joints["left_elbow"].y)),
                "left_elbow": calculate_angle((joints["right_shoulder"].x, joints["right_shoulder"].y),
                                               (joints["right_elbow"].x, joints["right_elbow"].y),
                                               (joints["right_wrist"].x, joints["right_wrist"].y)),
                "left_shoulder": calculate_angle((joints["right_hip"].x, joints["right_hip"].y),
                                                  (joints["right_shoulder"].x, joints["right_shoulder"].y),
                                                  (joints["right_elbow"].x, joints["right_elbow"].y))
            }

            # Convert normalized coordinates to pixels
            h, w, _ = frame.shape
            positions = {
                "right_elbow": (int(joints["left_elbow"].x * w), int(joints["left_elbow"].y * h)),
                "right_shoulder": (int(joints["left_shoulder"].x * w), int(joints["left_shoulder"].y * h)),
                "left_elbow": (int(joints["right_elbow"].x * w), int(joints["right_elbow"].y * h)),
                "left_shoulder": (int(joints["right_shoulder"].x * w), int(joints["right_shoulder"].y * h))
            }

            # Draw angles on the frame with yellow color and markers
            for joint, angle in angles.items():
                position = positions[joint]
                cv2.putText(frame, f"{int(angle)}°", position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)  # Yellow (0, 255, 255)

            # Add markers to indicate left and right sides (Swapped)
            cv2.putText(frame, "Left Side", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Right Side", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame in Streamlit
        frame_holder.image(frame, channels="BGR", use_container_width=True)

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()
