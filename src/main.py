from ultralytics import YOLO
import cv2 as cv
import numpy as np
import math

model = YOLO("yolo26s.pt")

cap = cv.VideoCapture("./data/test_video.mp4")
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_count += 1
    if success:
        results = model(frame, conf=0.25)
        for result in results:
            frame = result.plot()

        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()