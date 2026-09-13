from ultralytics import YOLO
import cv2 as cv
import tracking
import kalman
import forecasting
import evaluation 

model = YOLO("yolo26s.pt")

cap = cv.VideoCapture("./data/test_video.mp4")
frame_count = 0
history = {}
kalman_filters = {}
forecasts = {}

ades = []
fdes = []

while cap.isOpened():
    success, frame = cap.read()
    if success:
        #Run object detection and tracking  
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25)

        #Extract detection results
        name = [result[0].names[cls.item()] for cls in result[0].boxes.cls]
        track_id = result[0].boxes.id.numpy()
        box = result[0].boxes.xyxy.numpy()
        frame_count += 1
        frame = result[0].plot()
        for detection in zip(track_id, box, name):
            curr_track_id = detection[0]
            if curr_track_id not in forecasts:
                forecasts[curr_track_id] = []

            # Estimate the filtered current position and velocity using a Kalman filter
            filtered_position, velocity = kalman.filter(curr_track_id, detection[1], kalman_filters)

            x_center = (detection[1][0] + detection[1][2]) / 2
            y_center = (detection[1][1] + detection[1][3]) / 2
            centre = (x_center, y_center)

            # Track the trajectory history of the object and visualize it on the frame
            tracking.trajectory_history(curr_track_id, filtered_position, history, frame)

            #Evaluate the forecasted trajectory against the actual position of the object
            completed = evaluation.calculate_error(forecasts[curr_track_id], frame_count, centre)

            for ade, fde in completed:
                ades.append(ade)
                fdes.append(fde)

            # Generate and visualize a 10-frame constant-velocity forecast
            future_positions = forecasting.trajectory_forecast(filtered_position, velocity, frame, frame_count)
            new_forecast = {"predictions": future_positions, "errors": []}
            forecasts[curr_track_id].append(new_forecast)

        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()

print("\n--- Evaluation ---")

if len(ades) > 0:
    mean_ade = sum(ades) / len(ades)
    print("Mean ADE@10:", mean_ade)
    print("Evaluated forecasts:", len(ades))
else:
    print("No complete forecasts for ADE.")

if len(fdes) > 0:
    mean_fde = sum(fdes) / len(fdes)
    print("Mean FDE@10:", mean_fde)
else:
    print("No complete forecasts for FDE.")