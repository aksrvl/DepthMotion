import cv2 as cv

def trajectory_forecast(position, velocity, frame, curr_frame):
    """
    Predict the object's 2D trajectory for the next 10 frames
    using a constant-velocity motion model

    Args:
        position: Current filtered (x, y) position
        velocity:Estimated (vx, vy) velocity
        frame: Current video frame for visualization
        curr_frame: Index of the current video frame

    Returns:
        List of (frame_index, predicted_x, predicted_y) tuples
    """
    future_positions = []
    curr_x = position[0]
    curr_y = position[1]
    v_x = velocity[0]
    v_y = velocity[1]
    for n in range(1, 11):
        future_x = curr_x + n*v_x
        future_y = curr_y + n*v_y
        future_positions.append((curr_frame+n, future_x, future_y))
        cv.circle(frame, (int(future_x), int(future_y)), 5, (255, 0, 0), -1)
    return future_positions