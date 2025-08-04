from math import degrees, acos
import numpy as np

def calculate_angle(a, b, c):
    """Calculate angle at point b formed by points a-b-c"""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return degrees(acos(np.clip(cos_angle, -1.0, 1.0)))

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

def rate_form(sequence):
    """
    Rate pushup form based on multiple metrics:
    1. Body alignment (straight line from head to ankles)
    2. Elbow angle at bottom position (should be around 45-90 degrees)
    3. Range of motion (depth consistency)
    4. Hand position relative to shoulders
    """
    if not sequence:
        return 0
    
    scores = []
    
    for keypoints in sequence:
        if not keypoints or len(keypoints) < 33:
            continue
            
        # MediaPipe pose landmarks indices
        nose = keypoints[0]
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_elbow = keypoints[13]
        right_elbow = keypoints[14]
        left_wrist = keypoints[15]
        right_wrist = keypoints[16]
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        left_ankle = keypoints[27]
        right_ankle = keypoints[28]
        
        frame_score = 0
        total_metrics = 0
        
        # 1. Body alignment score (40% of total score)
        try:
            # Calculate alignment using shoulder, hip, and ankle
            shoulder_center = ((left_shoulder[0] + right_shoulder[0])/2, 
                             (left_shoulder[1] + right_shoulder[1])/2)
            hip_center = ((left_hip[0] + right_hip[0])/2, 
                         (left_hip[1] + right_hip[1])/2)
            ankle_center = ((left_ankle[0] + right_ankle[0])/2, 
                           (left_ankle[1] + right_ankle[1])/2)
            
            # Check if body forms a straight line (measure deviation)
            # Calculate angle at hip (should be close to 180 degrees for straight line)
            body_angle = calculate_angle(shoulder_center, hip_center, ankle_center)
            alignment_score = max(0, 10 - abs(body_angle - 180) / 2)  # More lenient
            frame_score += alignment_score * 0.4
            total_metrics += 0.4
        except:
            pass
        
        # 2. Elbow angle score (30% of total score)
        try:
            # Use the more visible elbow (typically right elbow in most videos)
            elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Good pushup form: elbow angle between 45-90 degrees at bottom
            if 45 <= elbow_angle <= 100:
                elbow_score = 10
            elif 40 <= elbow_angle <= 110:
                elbow_score = 8
            elif 35 <= elbow_angle <= 120:
                elbow_score = 6
            else:
                elbow_score = max(0, 10 - abs(elbow_angle - 70) / 10)
            
            frame_score += elbow_score * 0.3
            total_metrics += 0.3
        except:
            pass
        
        # 3. Hand position score (20% of total score)
        try:
            # Hands should be roughly shoulder-width apart and aligned with shoulders
            hand_distance = calculate_distance(left_wrist, right_wrist)
            shoulder_distance = calculate_distance(left_shoulder, right_shoulder)
            
            # Ideal hand distance is 1.2-1.5x shoulder width
            ideal_ratio = hand_distance / (shoulder_distance + 1e-7)
            if 1.0 <= ideal_ratio <= 1.8:
                hand_score = 10
            elif 0.8 <= ideal_ratio <= 2.0:
                hand_score = 7
            else:
                hand_score = max(0, 10 - abs(ideal_ratio - 1.3) * 5)
            
            frame_score += hand_score * 0.2
            total_metrics += 0.2
        except:
            pass
        
        # 4. Head position score (10% of total score)
        try:
            # Head should be in neutral position (not dropping or lifting too much)
            neck_angle = calculate_angle(shoulder_center, 
                                       ((left_shoulder[0] + right_shoulder[0])/2,
                                        (left_shoulder[1] + right_shoulder[1])/2), 
                                       nose)
            
            # Neutral head position should have neck angle around 160-180 degrees
            if 150 <= neck_angle <= 180:
                head_score = 10
            elif 140 <= neck_angle <= 190:
                head_score = 7
            else:
                head_score = max(0, 10 - abs(neck_angle - 165) / 10)
            
            frame_score += head_score * 0.1
            total_metrics += 0.1
        except:
            pass
        
        # Normalize score based on available metrics
        if total_metrics > 0:
            normalized_score = frame_score / total_metrics * 10
            scores.append(min(10, max(0, normalized_score)))
    
    return round(sum(scores) / len(scores), 2) if scores else 0