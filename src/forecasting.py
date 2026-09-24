def trajectory_forecast(X, Z, velocity, curr_frame):
    """
    Predict the object's BEV trajectory for the next 10 frames
    using a constant-velocity motion model.

    The prediction starts from the current Kalman-filtered BEV position
    (X, Z) and uses the estimated velocity (vX, vZ).

    Args:
        X: Current filtered lateral position in meters.
        Z: Current filtered forward depth in meters.
        velocity: Estimated (vX, vZ) velocity in meters per frame.
        curr_frame: Index of the current frame.

    Returns:
        List of (frame_index, predicted_X, predicted_Z) tuples
        for the next 10 frames.
    """
    future_positions = []
    curr_X = X
    curr_Z = Z
    v_X = velocity[0]
    v_Z = velocity[1]
    for n in range(1, 11):
        future_X = curr_X + n*v_X
        future_Z = curr_Z+ n*v_Z
        future_positions.append((curr_frame+n, float(future_X), float(future_Z)))
    return future_positions