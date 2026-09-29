import cv2 as cv
import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "src")
)

import tracking

tracking_model = tracking.load_tracking_model()

gt_trajectories = {}
gt_boxes = {}

files_RGB = os.listdir("./data/Camera_0_RGB")
files_RGB.sort()

with open ('./data/pose.txt', 'r') as file_pose:
    pose_lines = file_pose.readlines()

for line in pose_lines[1:]:
    values = line.split()
    if values[1] == "0":
        frame = int(values[0])
        track_id = int(values[2])
        camera_X = float(values[13])
        camera_Z = float(values[15])
        if track_id not in gt_trajectories:
            gt_trajectories[track_id] = []
        gt_trajectories[track_id].append((frame, camera_X, camera_Z))

with open('./data/bbox.txt', 'r') as file_bbox:
    bbox_lines = file_bbox.readlines()

for line in bbox_lines[1:]:
    values = line.split()
    if values[1] == "0":
        frame = int(values[0])
        track_id = int(values[2])
        left = int(values[3])
        right = int(values[4])
        top = int(values[5])
        bottom = int(values[6])
        if frame not in gt_boxes:
            gt_boxes[frame] = []
        gt_boxes[frame].append((track_id, [left, top, right, bottom]))


def calculate_iou(box1, box2):
    x1_intersect = max(box1[0], box2[0])
    y1_intersect = max(box1[1], box2[1])
    x2_intersect = min(box1[2], box2[2])
    y2_intersect = min(box1[3], box2[3])

    height_intersect = max(0, y2_intersect - y1_intersect)
    width_intersect = max(0, x2_intersect - x1_intersect)

    area_intersect = width_intersect * height_intersect

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - area_intersect

    iou = area_intersect / union

    return iou

match_history = []

for curr_frame, image_file in enumerate(files_RGB):
    image = cv.imread("./data/Camera_0_RGB/" + image_file)
    tracked_objects = tracking.track_objects(image, tracking_model)

    pairs = []
    for (bytetrack_id, predicted_box) in tracked_objects:
        for (gt_id, gt_box) in gt_boxes.get(curr_frame, []):
            curr_iou = calculate_iou(predicted_box, gt_box)
            if curr_iou >= 0.5:
                pairs.append((bytetrack_id, gt_id, curr_iou))

    pairs.sort(key=lambda pair: pair[2], reverse=True)
    used_bt_ids = []
    used_gt_ids = []
    for (bytetrack_id, gt_id, curr_iou) in pairs:
        if bytetrack_id not in used_bt_ids and gt_id not in used_gt_ids:
            match_history.append((curr_frame, int(bytetrack_id), gt_id, float(curr_iou)))
            used_bt_ids.append(bytetrack_id)
            used_gt_ids.append(gt_id)

print("Total matches:", len(match_history))
print(match_history[:20])
print("RGB frames:", len(files_RGB))
print("GT frames:", len(gt_boxes))
print("Last RGB index:", len(files_RGB) - 1)
print("Last GT frame:", max(gt_boxes.keys()))