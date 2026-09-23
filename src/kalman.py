import cv2 as cv
import numpy as np

def filter(track_id, X, Z, filters):
    """
    Estimate the filtered BEV position and velocity of a tracked object
    using a constant-velocity Kalman filter.

    The state vector is [X, Z, vX, vZ], where X is the lateral position
    and Z is the forward depth in meters. The measurement contains the
    estimated BEV position [X, Z]. A separate Kalman filter is maintained
    for each track ID.

    Args:
        track_id: Persistent ByteTrack ID of the object.
        X: Estimated lateral position in meters.
        Z: Estimated forward depth in meters.
        filters: Dictionary containing one Kalman filter per track ID.

    Returns:
        filtered_position: Estimated (X, Z) position in meters.
        velocity: Estimated (vX, vZ) velocity in meters per frame.
    """
    if track_id not in filters:
        kf = cv.KalmanFilter(4 ,2)

        # State: [X, Z, vX, vZ], measurement: [X, Z]
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
        kf.statePost = np.array([[X], [Z], [0], [0]], np.float32)
        filters[track_id] = kf
        return (X, Z), (0, 0)

    else:
        kf = filters[track_id]
        measurement = np.array([[X], [Z]], np.float32)
        kf.predict()
        kf.correct(measurement)
        state_post = kf.statePost
        X_filtered = state_post[0][0]
        Z_filtered = state_post[1][0]
        vX = state_post[2][0]
        vZ = state_post[3][0]

    return (X_filtered, Z_filtered), (vX, vZ)


