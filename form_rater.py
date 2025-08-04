# from exercise_metrics import (
#     pushups,
#     bicep_curls,
#     squats,
#     pullups,
#     lat_pulldown,
#     dips,
#     bench_press,
#     deadlifts,
#     lunges,
#     plank,
#     mountain_climbers,
#     russian_twists,
#     hanging_leg_raises,
#     shoulder_press,
#     chest_fly,
#     glute_bridges,
#     calf_raises,
#     hip_thrusts,
#     romanian_deadlifts,
#     sumo_deadlifts,
#     bulgarian_split_squats,
#     lateral_raises,
#     front_raises,
#     incline_bench_press,
#     tricep_extensions,
#     seated_row,
#     barbell_row,
#     power_clean,
#     clean_and_press,
#     snatch
# )

# RATER_FUNCTIONS = {
#     "pushups": pushups.rate_form,
#     "bicep_curls": bicep_curls.rate_form,
#     "squats": squats.rate_form,
#     "pullups": pullups.rate_form,
#     "lat_pulldown": lat_pulldown.rate_form,
#     "dips": dips.rate_form,
#     "bench_press": bench_press.rate_form,
#     "deadlifts": deadlifts.rate_form,
#     "lunges": lunges.rate_form,
#     "plank": plank.rate_form,
#     "mountain_climbers": mountain_climbers.rate_form,
#     "russian_twists": russian_twists.rate_form,
#     "hanging_leg_raises": hanging_leg_raises.rate_form,
#     "shoulder_press": shoulder_press.rate_form,
#     "chest_fly": chest_fly.rate_form,
#     "glute_bridges": glute_bridges.rate_form,
#     "calf_raises": calf_raises.rate_form,
#     "hip_thrusts": hip_thrusts.rate_form,
#     "romanian_deadlifts": romanian_deadlifts.rate_form,
#     "sumo_deadlifts": sumo_deadlifts.rate_form,
#     "bulgarian_split_squats": bulgarian_split_squats.rate_form,
#     "lateral_raises": lateral_raises.rate_form,
#     "front_raises": front_raises.rate_form,
#     "incline_bench_press": incline_bench_press.rate_form,
#     "tricep_extensions": tricep_extensions.rate_form,
#     "seated_row": seated_row.rate_form,
#     "barbell_row": barbell_row.rate_form,
#     "power_clean": power_clean.rate_form,
#     "clean_and_press": clean_and_press.rate_form,
#     "snatch": snatch.rate_form
# }

#above section commented out to test pushups.py implemntation only

from exercise_metrics.pushups import rate_form

RATER_FUNCTIONS = {
    'pushups': rate_form,
}

def rate_exercise(exercise_name, keypoints_sequence):
    """
    Rate exercise form based on keypoints sequence
    
    Args:
        exercise_name (str): Name of the exercise
        keypoints_sequence (list): List of keypoints for each frame
    
    Returns:
        float: Form rating score (0-10)
    """
    exercise_name = exercise_name.lower().replace(' ', '_').replace('-', '_')
    
    if exercise_name not in RATER_FUNCTIONS:
        available_exercises = list(RATER_FUNCTIONS.keys())
        raise ValueError(f"Unsupported exercise: {exercise_name}. "
                        f"Available exercises: {', '.join(available_exercises)}")
    
    # Filter out None keypoints before rating
    valid_keypoints = [kp for kp in keypoints_sequence if kp is not None]
    
    if not valid_keypoints:
        raise ValueError("No valid keypoints found in the sequence")
    
    rating = RATER_FUNCTIONS[exercise_name](valid_keypoints)
    
    # Ensure rating is within bounds
    return max(0, min(10, rating))
