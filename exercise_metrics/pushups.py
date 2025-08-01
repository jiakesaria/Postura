from math import degrees, acos
import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return degrees(acos(np.clip(cos_angle, -1.0, 1.0)))

def rate_form(sequence):
    scores = []
    for keypoints in sequence:
        if not keypoints: continue
        shoulder = keypoints[12]
        elbow = keypoints[14]
        wrist = keypoints[16]
        angle = calculate_angle(shoulder, elbow, wrist)
        score = max(0, min(10 - abs(angle - 90) / 9, 10))
        scores.append(score)
    return round(sum(scores) / len(scores), 2) if scores else 0
