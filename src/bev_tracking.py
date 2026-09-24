import cv2 as cv
from ultralytics import YOLO
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt
import kalman
import forecasting

track_history = {}
filters = {}
filtered_track_history = {}
forecasts = {}

with open ('./data/intrinsic.txt', 'r') as file:
    lines = file.readlines()

values = lines[1].split()
frame_id = int(values[0])
camera_id = int(values[1])

fx = float(values[2])
fy = float(values[3])
cx = float(values[4])
cy = float(values[5])

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "third_party")
)

from depth_anything_v2.dpt import DepthAnythingV2

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}

max_depth = 80

depth_model = DepthAnythingV2(**{**model_config, 'max_depth': max_depth})
depth_model.load_state_dict(torch.load('models/depth_anything_v2_metric_vkitti_vits.pth', map_location='cpu'))
depth_model = depth_model.to(device)
depth_model.eval()

model = YOLO("yolo26n.pt")

files_RGB = os.listdir("./data/Camera_0_RGB")
files_RGB.sort()

for curr_frame, image_file in enumerate(files_RGB):

    frame = cv.imread("./data/Camera_0_RGB/" + image_file)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)

    depth = depth_model.infer_image(frame)

    for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            if result.boxes.id is None:
                continue

            track_ids = result.boxes.id.cpu().numpy()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
    
                x_1 = int(x1 + (x2 - x1)*0.3)
                x_2 = int(x2 - (x2 - x1)*0.3)
                y_1 = int(y1 + (y2 - y1)*0.3)
                y_2 = int(y2 - (y2 - y1)*0.3)
    
                centre_region = depth[y_1:y_2, x_1:x_2]
                pred_depth = np.median(centre_region)

                u = int((x1 +x2)/2)
                v = int(y2)
                Z = pred_depth
                X = (u - cx)*(Z/fx)

                filtered_position, velocity = kalman.filter(track_id, X, Z, filters)
                X_filtered, Z_filtered = filtered_position

                future_positions = forecasting.trajectory_forecast(X_filtered, Z_filtered, velocity, curr_frame)

                forecasts[track_id] = future_positions

                if track_id not in track_history:
                    track_history[track_id] = []

                if track_id not in filtered_track_history:
                    filtered_track_history[track_id] = []

                track_history[track_id].append((float(X), float(Z)))
                filtered_track_history[track_id].append((float(X_filtered), float(Z_filtered)))

print(forecasts[5])

# Compare raw and Kalman-filtered BEV trajectory for a single track
track_to_plot = 5

raw_X = []
raw_Z = []

filtered_X = []
filtered_Z = []

forecast_X = []
forecast_Z = []

for X, Z in track_history[track_to_plot]:
    raw_X.append(X)
    raw_Z.append(Z)

for X, Z in filtered_track_history[track_to_plot]:
    filtered_X.append(X)
    filtered_Z.append(Z)

for frame_idx, X, Z in forecasts[track_to_plot]:
    forecast_X.append(X)
    forecast_Z.append(Z)

plt.figure(figsize=(10, 8))

plt.plot(
    raw_X,
    raw_Z,
    marker="o",
    markersize=5,
    linewidth=1.5,
    label="Raw depth trajectory"
)

plt.plot(
    filtered_X,
    filtered_Z,
    marker="o",
    markersize=4,
    linewidth=2,
    label="Kalman filtered trajectory"
)

plt.plot(
    forecast_X,
    forecast_Z,
    marker="x",
    markersize=6,
    linewidth=2,
    linestyle="--",
    label="10-frame forecast"
)

plt.scatter(
    0,
    0,
    marker="^",
    s=120,
    label="Camera"
)

plt.xlim(-15, 15)
plt.ylim(0, 80)

plt.xlabel("X Position (m)")
plt.ylabel("Depth Z (m)")
plt.title(f"Raw vs Kalman-Filtered BEV Trajectory — Track ID {track_to_plot}")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
