import tracking
import kalman
import forecasting
import depth
import geometry
import cv2 as cv
import os

tracking_model = tracking.load_tracking_model()
depth_model = depth.load_depth_model()

fx, fy, cx, cy = geometry.load_intrinsics()

track_history = {}
kalman_filters = {}
forecasts = {}

files_RGB = os.listdir("./data/Camera_0_RGB")
files_RGB.sort()

for curr_frame, image_file in enumerate(files_RGB):
    frame = cv.imread("./data/Camera_0_RGB/" + image_file)
    tracked_objects = tracking.track_objects(frame, tracking_model)
    depth_map = depth.depth_map(frame, depth_model)

    for track_id, box in tracked_objects:
        X, Z = geometry.bbox_to_bev(box, depth_map, fx, cx)
        filtered_position, velocity = kalman.filter(track_id, X, Z, kalman_filters)
        X_filtered, Z_filtered = filtered_position

        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((float(X_filtered), float(Z_filtered)))

        future_positions = forecasting.trajectory_forecast(X_filtered, Z_filtered, velocity, curr_frame)
        forecasts[track_id] = future_positions

print(track_history[5])
print(forecasts[5])