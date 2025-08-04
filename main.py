import argparse
import os
import sys
from video_utils import extract_frames
from keypoint_extractor import extract_keypoints
from form_rater import rate_exercise

def main():
    parser = argparse.ArgumentParser(description="Postura: Exercise Form Evaluator")
    parser.add_argument("--exercise", type=str, required=True, 
                       help="Name of the exercise (e.g., pushups)")
    parser.add_argument("--video", type=str, required=True, 
                       help="Path to the exercise video")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        sys.exit(1)
    
    exercise_name = args.exercise.lower().strip()
    
    try:
        print(f"Analyzing {exercise_name} form from video: {args.video}")
        print("-" * 50)
        
        # Extract frames from video
        print("Step 1: Extracting frames from video...")
        frames = extract_frames(args.video)
        
        if not frames:
            print("Error: No frames could be extracted from the video.")
            sys.exit(1)
        
        # Extract pose keypoints
        print("Step 2: Extracting pose keypoints...")
        keypoints_seq = extract_keypoints(frames)
        
        # Filter out None keypoints and count valid frames
        valid_keypoints = [kp for kp in keypoints_seq if kp is not None]
        
        if not valid_keypoints:
            print("Error: No valid pose keypoints detected in the video.")
            print("Tips:")
            print("- Ensure the person is clearly visible in the video")
            print("- Make sure the lighting is adequate")
            print("- The person should be facing the camera")
            sys.exit(1)
        
        print(f"Valid pose data found in {len(valid_keypoints)}/{len(frames)} frames")
        
        # Rate the exercise form
        print("Step 3: Analyzing exercise form...")
        rating = rate_exercise(exercise_name, keypoints_seq)
        
        print("-" * 50)
        print(f"{exercise_name.title()} Form Rating: {rating}/10")
        
        # Provide feedback based on rating
        if rating >= 8.5:
            print("Excellent form! Keep up the great work!")
        elif rating >= 7.0:
            print("Good form with minor areas for improvement.")
        elif rating >= 5.5:
            print("Moderate form. Focus on proper technique.")
        else:
            print(" Poor form detected. Consider reviewing proper technique.")
        
        print("-" * 50)
        
        if args.verbose:
            print(f"Analysis completed on {len(valid_keypoints)} frames")
            print(f"Frame extraction rate: Every 3rd frame")
            print(f"Pose detection success rate: {len(valid_keypoints)/len(frames)*100:.1f}%")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your video file and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()