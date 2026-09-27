import tracking
import kalman
import forecasting
import depth
import geometry
import visualization
import cv2 as cv
import os
import numpy as np

tracking_model = tracking.load_tracking_model()
depth_model = depth.load_depth_model()

fx, fy, cx, cy = geometry.load_intrinsics()

track_history = {}
kalman_filters = {}
forecasts = {}
last_seen = {}

MAX_MISSING_FRAMES = 6

writer = None

files_RGB = os.listdir("./data/Camera_0_RGB")
files_RGB.sort()

for curr_frame, image_file in enumerate(files_RGB):
    frame = cv.imread("./data/Camera_0_RGB/" + image_file)

    tracked_objects = tracking.track_objects(frame, tracking_model)
    depth_map = depth.depth_map(frame, depth_model)
    
    for track_id, box in tracked_objects:
        last_seen[track_id] = curr_frame

        X, Z = geometry.bbox_to_bev(box, depth_map, fx, cx)

        filtered_position, velocity = kalman.filter(track_id, X, Z, kalman_filters)

        X_filtered, Z_filtered = filtered_position

        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((float(X_filtered), float(Z_filtered)))

        future_positions = forecasting.trajectory_forecast(X_filtered, Z_filtered, velocity, curr_frame)
        forecasts[track_id] = future_positions

    frame = visualization.draw_detections(
        frame,
        tracked_objects
    )

    inactive_tracks = [
        track_id
        for track_id, last_frame in last_seen.items()
        if curr_frame - last_frame > MAX_MISSING_FRAMES
    ]

    for track_id in inactive_tracks:
        track_history.pop(track_id, None)
        forecasts.pop(track_id, None)
        kalman_filters.pop(track_id, None)
        last_seen.pop(track_id, None)

    bev_image = visualization.draw_bev(
        track_history,
        forecasts
    )

    frame_width = frame.shape[1]
    bev_height = 500

    bev_image = cv.resize(
        bev_image,
        (frame_width, bev_height)
    )
    combined = np.vstack((frame, bev_image))

    if writer is None:
        height, width = combined.shape[:2]

        writer = cv.VideoWriter(
            "results/depthmotion_demo.mp4",
            cv.VideoWriter_fourcc(*"mp4v"),
            10,
            (width, height)
        )

    writer.write(combined)

    cv.imshow("DepthMotion", combined)

    if cv.waitKey(1) == ord("q"):
        break

if writer is not None:
    writer.release()

cv.destroyAllWindows()
