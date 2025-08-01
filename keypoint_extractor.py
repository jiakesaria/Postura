import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

def extract_keypoints(frames):
    pose_keypoints = []
    with mp_pose.Pose(static_image_mode=False) as pose:
        for frame in frames:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            if result.pose_landmarks:
                pose_keypoints.append([(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark])
            else:
                pose_keypoints.append(None)
    return pose_keypoints
