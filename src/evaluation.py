from math import sqrt

def calculate_error(forecasts, curr_frame, centre):
    """
    Evaluate stored trajectory forecasts against the current observed position.

    For every prediction targeting the current frame, the function computes
    the Euclidean position error. Once a 10-frame forecast is complete, 
    ADE and FDE are returned.

    Args:
        forecasts: List of stored forecasts for one track ID
        curr_frame: Index of the current video frame
        centre: Observed bounding-box center (x, y) in pixels

    Returns:
        List of (ADE, FDE) tuples for completed forecasts on this frame.
    """
    completed = []
    for forecast in forecasts:
        for prediction in forecast["predictions"]:
            if prediction[0] == curr_frame:
                predicted_x = prediction[1]
                predicted_y = prediction[2]

                dx = predicted_x - centre[0]
                dy = predicted_y - centre[1]

                error = sqrt(dx**2 + dy**2)

                forecast["errors"].append(error)

                if len(forecast["errors"]) == 10:
                    ade = sum(forecast["errors"]) / 10
                    fde = forecast["errors"][-1]

                    completed.append((ade, fde))

    return completed