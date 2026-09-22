import cv2 as cv
import numpy as np
from ultralytics import YOLO
import sys
import os
import torch
import math
from matplotlib import pyplot as plt

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

files_depth = os.listdir("./data/Camera_0_depth")
files_RGB = os.listdir("./data/Camera_0_RGB")

files_depth.sort()
files_RGB.sort()


evaluation_results = []
for image_file, depth_file in zip(files_RGB, files_depth):

    frame = cv.imread("./data/Camera_0_RGB/" + image_file)
    results = model(frame)
    depth = depth_model.infer_image(frame)

    gt_depth = cv.imread("./data/Camera_0_depth/"+depth_file, cv.IMREAD_UNCHANGED)
    gt_depth_m = gt_depth.astype(np.float32) / 100.0
    valid_mask = gt_depth != 65535

    # print(image_file)
    # print(depth_file)
    # print(gt_depth_m.shape)
    # print(gt_depth_m.dtype)
    # print(gt_depth_m.min(), gt_depth_m.max())
    # print(depth.shape)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box

            x_1 = int(x1 + (x2 - x1)*0.3)
            x_2 = int(x2 - (x2 - x1)*0.3)
            y_1 = int(y1 + (y2 - y1)*0.3)
            y_2 = int(y2 - (y2 - y1)*0.3)

            pred_region= depth[y_1:y_2, x_1:x_2]
            gt_region=gt_depth_m[y_1:y_2, x_1:x_2]
            region_valid_mask = valid_mask[y_1:y_2, x_1:x_2]
            gt_valid_vals = gt_region[region_valid_mask]

            pred_depth = np.median(pred_region)
            gt_depth_value = np.median(gt_valid_vals)
            absolute_error = abs(pred_depth - gt_depth_value)

            # print("Predicted:", pred_depth)
            # print("GT:", gt_depth_value)
            # print("Absolute error:", absolute_error)
            evaluation_results.append([float(pred_depth), float(gt_depth_value), float(absolute_error)])

MAE = 0
RMSE = 0
AbsRel = 0
for evaluation in evaluation_results:
    MAE += evaluation[2]
    RMSE += (evaluation[0] - evaluation[1])**2
    AbsRel += evaluation[2]/evaluation[1]
    print(evaluation)

MAE /= len(evaluation_results)
RMSE = math.sqrt(RMSE/len(evaluation_results))
AbsRel = AbsRel / len(evaluation_results) * 100
# print("MAE = ", MAE, "m")
# print("RMSE = ", RMSE, 'm')
# print("AbsRel = ", AbsRel, "%")

depth_range_errors = {"0-10": [], 
                      "10-20": [], 
                      "20-30": [], 
                      "30-40": [], 
                      "40-50": [], 
                      "50-60": [], 
                      "60-80": []}

for evaluation in evaluation_results:
    if evaluation[1] < 10:
        depth_range_errors["0-10"].append(evaluation[2])
    elif evaluation[1] < 20:
            depth_range_errors["10-20"].append(evaluation[2])
    elif evaluation[1] < 30:
            depth_range_errors["20-30"].append(evaluation[2])
    elif evaluation[1] < 40:
            depth_range_errors["30-40"].append(evaluation[2])
    elif evaluation[1] < 50:
            depth_range_errors["40-50"].append(evaluation[2])
    elif evaluation[1] < 60:
            depth_range_errors["50-60"].append(evaluation[2])
    elif evaluation[1] < 80:
          depth_range_errors["60-80"].append(evaluation[2])

distance_ranges = []
mae_by_distance = []
for key in depth_range_errors:
    range_mae = sum(depth_range_errors[key])/len(depth_range_errors[key])
    distance_ranges.append(key)
    mae_by_distance.append(range_mae)
    print(key, "m: N = ", len(depth_range_errors[key]), "MAE = ", range_mae, "m")

plt.plot(distance_ranges, mae_by_distance, marker="o")
plt.xlabel("Ground Truth Distance (m)")
plt.ylabel("MAE (m)")
plt.title("Depth Estimation Error vs. Distance")
plt.grid(visible=True)
plt.show()