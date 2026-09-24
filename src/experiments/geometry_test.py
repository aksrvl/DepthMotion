import cv2 as cv
from ultralytics import YOLO
import numpy as np
import sys
import os
import torch

model = YOLO("yolo26n.pt")
results = model('./data/rgb_00000.jpg')
results[0].show()

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

with open ('./data/intrinsic.txt', 'r') as file:
    lines = file.readlines()

print(lines[0])
print(lines[1])

values = lines[1].split()
frame_id = int(values[0])
camera_id = int(values[1])

fx = float(values[2])
fy = float(values[3])
cx = float(values[4])
cy = float(values[5])

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = box
        u = int((x1 +x2)/2)
        v = int(y2)
        x_1 = int(x1 + (x2 - x1)*0.3)
        x_2 = int(x2 - (x2 - x1)*0.3)
        y_1 = int(y1 + (y2 - y1)*0.3)
        y_2 = int(y2 - (y2 - y1)*0.3)
        centre_region = depth[y_1:y_2, x_1:x_2]
        median_depth = np.median(centre_region)
        Z = median_depth
        print("u: ",u, "v: ", v, "depth: ", median_depth)

        X = (u - cx)*(Z/fx)
        Y = (v - cy)*(Z/fy)

        print("X=", X, "Y=", Y)

