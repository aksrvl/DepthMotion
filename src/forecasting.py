import cv2 as cv

def trajectory_forecast(position, velocity, frame, curr_frame):
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