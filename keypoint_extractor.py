import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose

def extract_keypoints(frames):
    """
    Extract pose keypoints with improved confidence filtering and smoothing
    """
    pose_keypoints = []
    
    # Configure MediaPipe with optimized settings
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Balance between accuracy and speed
        smooth_landmarks=True,  # Enable landmark smoothing
        enable_segmentation=False,  # Disable segmentation for speed
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        for i, frame in enumerate(frames):
            # Enhance frame for better detection
            enhanced_frame = enhance_frame_for_pose_detection(frame)
            image_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            
            result = pose.process(image_rgb)
            
            if result.pose_landmarks:
                # Extract landmarks with confidence filtering
                landmarks = []
                for lm in result.pose_landmarks.landmark:
                    # Only include landmarks with reasonable visibility
                    if lm.visibility > 0.5:
                        landmarks.append((lm.x, lm.y, lm.z, lm.visibility))
                    else:
                        landmarks.append((0, 0, 0, 0))  # Placeholder for low confidence points
                
                pose_keypoints.append(landmarks)
            else:
                pose_keypoints.append(None)
                print(f"Warning: No pose detected in frame {i}")
    
    # Apply temporal smoothing to reduce jitter
    smoothed_keypoints = smooth_keypoints_sequence(pose_keypoints)
    
    print(f"Successfully extracted keypoints from {len([kp for kp in smoothed_keypoints if kp is not None])}/{len(frames)} frames")
    return smoothed_keypoints

def enhance_frame_for_pose_detection(frame):
    """
    Enhance frame for better pose detection
    """
    # Increase contrast and brightness slightly
    enhanced = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def smooth_keypoints_sequence(keypoints_sequence, window_size=3):
    """
    Apply temporal smoothing to keypoints to reduce noise
    """
    if len(keypoints_sequence) < window_size:
        return keypoints_sequence
    
    smoothed = []
    for i, keypoints in enumerate(keypoints_sequence):
        if keypoints is None:
            smoothed.append(None)
            continue
        
        # Collect valid keypoints from neighboring frames
        valid_frames = []
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(keypoints_sequence), i + window_size // 2 + 1)
        
        for j in range(start_idx, end_idx):
            if keypoints_sequence[j] is not None:
                valid_frames.append(keypoints_sequence[j])
        
        if valid_frames:
            # Average the keypoints across valid frames
            smoothed_frame = []
            for landmark_idx in range(len(keypoints)):
                x_coords = [frame[landmark_idx][0] for frame in valid_frames if frame[landmark_idx][3] > 0.5]
                y_coords = [frame[landmark_idx][1] for frame in valid_frames if frame[landmark_idx][3] > 0.5]
                z_coords = [frame[landmark_idx][2] for frame in valid_frames if frame[landmark_idx][3] > 0.5]
                
                if x_coords:  # If we have valid coordinates
                    avg_x = np.mean(x_coords)
                    avg_y = np.mean(y_coords)
                    avg_z = np.mean(z_coords)
                    avg_vis = keypoints[landmark_idx][3]  # Keep original visibility
                    smoothed_frame.append((avg_x, avg_y, avg_z, avg_vis))
                else:
                    smoothed_frame.append(keypoints[landmark_idx])
            
            smoothed.append(smoothed_frame)
        else:
            smoothed.append(keypoints)
    
    return smoothed