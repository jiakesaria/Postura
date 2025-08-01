import argparse
from video_utils import extract_frames
from keypoint_extractor import extract_keypoints
from form_rater import rate_exercise

def main():
    parser = argparse.ArgumentParser(description="Postura: Exercise Form Evaluator")
    parser.add_argument("--exercise", type=str, required=True, help="Name of the exercise")
    parser.add_argument("--video", type=str, required=True, help="Path to the exercise video")
    args = parser.parse_args()

    frames = extract_frames(args.video)
    keypoints_seq = extract_keypoints(frames)
    
    rating = rate_exercise(args.exercise.lower(), keypoints_seq)
    print(f"{args.exercise.title()} form rating: {rating}/10")

if __name__ == "__main__":
    main()
