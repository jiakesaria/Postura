import cv2
import numpy as np

def extract_frames(video_path, every_n_frames=3):
    """
    Extract frames from video with better error handling and frame selection
    Reduced frame interval for better motion capture
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing video: {total_frames} frames at {fps:.2f} FPS")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % every_n_frames == 0:
            # Resize frame if it's too large (for faster processing)
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            frames.append(frame)
        count += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames for analysis")
    return frames

def enhance_frame_for_pose_detection(frame):
    """
    Enhance frame for better pose detection
    """
    # Increase contrast and brightness slightly
    enhanced = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced