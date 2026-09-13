import cv2 as cv

def trajectory_history(track_id, positions, history, frame):
    """
    Store and visualize the filtered 2D trajectory of a tracked object

    Args:
        track_id: Persistent ID of the tracked object
        positions: Current filtered (x, y) position
        history: Dictionary storing position history for each track ID
        frame: Current vifdeo frame for visualization
    """
    x_center = positions[0]
    y_center = positions[1]
    if track_id not in history:
        history[track_id] = []
    history[track_id].append((x_center, y_center))

    for track_id, positions in history.items():
        for i in range(0, len(positions)):
            cv.circle(frame, (int(positions[i][0]), int(positions[i][1])), 5, (0, 255, 0), -1)
    