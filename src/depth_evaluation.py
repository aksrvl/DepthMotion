import cv2 as cv
import numpy as np
from ultralytics import YOLO
import sys
import os
import torch

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

frame = cv.imread('./data/rgb_00000.jpg')
max_depth = 80

depth_model = DepthAnythingV2(**{**model_config, 'max_depth': max_depth})
depth_model.load_state_dict(torch.load('models/depth_anything_v2_metric_vkitti_vits.pth', map_location='cpu'))
depth_model = depth_model.to(device)
depth_model.eval()
depth = depth_model.infer_image(frame)

model = YOLO("yolo26n.pt")
results = model('./data/rgb_00000.jpg')
results[0].show()

gt_depth = cv.imread("./data/depth_00000.png", cv.IMREAD_UNCHANGED)
gt_depth_m = gt_depth.astype(np.float32) / 100.0
valid_mask = gt_depth != 65535

print(gt_depth_m.shape)
print(gt_depth_m.dtype)
print(gt_depth_m.min(), gt_depth_m.max())
print(depth.shape)

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

        print("Predicted:", pred_depth)
        print("GT:", gt_depth_value)
        print("Absolute error:", absolute_error)
