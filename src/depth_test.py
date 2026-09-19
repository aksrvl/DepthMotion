import sys
import os
import cv2 as cv
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
yolo_model = YOLO("yolo26s.pt")

# Make the copied Depth Anything V2 code importable
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

cap = cv.VideoCapture("./data/test_video.mp4")
success, frame = cap.read()
if success:
    depth = depth_model.infer_image(frame)
    results = yolo_model(frame)
    for result in results:
        name = [result.names[cls.item()] for cls in result.boxes.cls]
        box = result.boxes.xyxy.numpy()
        for detection in zip(name, box):
            centre_y = int((detection[1][1] + detection[1][3])/2)
            centre_x = int((detection[1][0] + detection[1][2])/2)

            x_1 = int(detection[1][0] + (detection[1][2] - detection[1][0])*0.3)
            x_2 = int(detection[1][2] - (detection[1][2] - detection[1][0])*0.3)
            y_1 = int(detection[1][1] + (detection[1][3] - detection[1][1])*0.3)
            y_2 = int(detection[1][3] - (detection[1][3] - detection[1][1])*0.3)
            region = depth[y_1:y_2, x_1:x_2]
            median_depth = np.median(region)
            pixel_depth = depth[centre_y, centre_x]

            cv.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)
            cv.circle(frame, (centre_x, centre_y), 10, (0, 0, 255), -1)
            text = f"{detection[0]} {median_depth:.1f} m"

            cv.putText(frame, text, (int(detection[1][0]), int(detection[1][1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
            print(f'1. {detection[0]} {pixel_depth} m')
            print(f'2. {detection[0]} {median_depth} m')
    
            
cap.release()


print(depth.shape)
print(depth.min(), depth.max())

plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
plt.show()