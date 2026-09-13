import cv2 as cv
import numpy as np

def filter(track_id, box, filters):
    """
    Estimate the filtered 2D position and velocity of a tracked object
    using a constant-velocity Kalman filter.

    The state vector is [x, y, vx, vy], while the measurement contains
    only the bounding-box centre [x, y]. A separate Kalman filter is
    maintained for each track ID.

    Args:
        track_id: Persistent ByteTrack ID of the object.
        box: Bounding box in (x1, y1, x2, y2) format.
        filters: Dictionary containing one Kalman filter per track ID.

    Returns:
        filtered_position: Estimated (x, y) position.
        velocity: Estimated (vx, vy) velocity.
    """
    x_center = (box[0] + box[2])/2
    y_center = (box[1] + box[3])/2
    if track_id not in filters:
        kf = cv.KalmanFilter(4 ,2)

        # State: [x, y, vx, vy], measurement: [x, y]
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # Constant-velocity motion model with dt = 1 frame
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        kf.measurementNoiseCov = np.array([[1, 0], [0,1]], np.float32) * 0.1
        kf.statePost = np.array([[x_center], [y_center], [0], [0]], np.float32)
        filters[track_id] = kf
        return (x_center, y_center), (0, 0)

    else:
        kf = filters[track_id]
        measurement = np.array([[x_center], [y_center]], np.float32)
        kf.predict()
        kf.correct(measurement)
        state_post = kf.statePost
        x_filtered = state_post[0][0]
        y_filtered = state_post[1][0]
        vx = state_post[2][0]
        vy = state_post[3][0]

    return (x_filtered, y_filtered), (vx, vy)


