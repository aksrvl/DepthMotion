import cv2 as cv
import matplotlib.pyplot as plt

def draw_detections(frame, tracked_objects):
    for track_id, box in tracked_objects:
        x1, y1, x2, y2 = map(int, box)
        cv.rectangle(frame, 
                     (x1, y1), 
                     (x2, y2), 
                     (0, 0, 255), 
                     3
                     )
        cv.putText(frame, 
                   f"ID{int(track_id)}", 
                   (x1, y1-10), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   0.6, 
                   (0, 0, 255), 
                   2
                   )
    return frame

def draw_bev(track_history, forecasts):
    """
    Visualize filtered object trajectories and future forecasts
    in Bird's-Eye View (BEV).

    Args:
        track_history: Dictionary containing filtered (X, Z)
            trajectory history for each track ID.
        forecasts: Dictionary containing future predicted (X, Z)
            positions for each track ID.
    """
    fig, ax = plt.subplots(figsize=(6,8))
    ax.set_xlim(-15, 15)
    ax.set_ylim(0, 80)

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Depth Z (m)")
    ax.set_title("DepthMotion — Bird's-Eye View")
    ax.grid(alpha=0.3)

    ax.scatter(
    0,
    0,
    marker="^",
    s=100,
    label="Camera"
    )   

    # Filtered trajectory history
    for track_id, positions in track_history.items():
        positions = positions[-20:]

        if len(positions) < 2:
            continue

        X_history = [position[0] for position in positions]
        Z_history = [position[1] for position in positions]

        ax.plot(
            X_history,
            Z_history,
            marker="o",
            markersize=3,
            label=f"ID {int(track_id)}"
        )

    track_colors = {}

    # Filtered trajectory history
    for track_id, positions in track_history.items():
        positions = positions[-20:]

        if len(positions) < 2:
            continue

        X_history = [position[0] for position in positions]
        Z_history = [position[1] for position in positions]

        line, = ax.plot(
            X_history,
            Z_history,
            marker="o",
            markersize=3
        )

        # Remember the color assigned by matplotlib
        track_colors[track_id] = line.get_color()

        # Write ID next to the current position
        ax.text(
            X_history[-1],
            Z_history[-1],
            f" ID {int(track_id)}",
            color=line.get_color(),
            fontsize=8,
            clip_on=True
        )


    # Future trajectory forecasts
    for track_id, future_positions in forecasts.items():

        if track_id not in track_history:
            continue

        if track_id not in track_colors:
            continue

        current_X, current_Z = track_history[track_id][-1]

        forecast_X = [current_X]
        forecast_Z = [current_Z]

        for _, future_X, future_Z in future_positions:
            forecast_X.append(future_X)
            forecast_Z.append(future_Z)

        ax.plot(
            forecast_X,
            forecast_Z,
            linestyle="--",
            marker="x",
            markersize=4,
            color=track_colors[track_id]
        )

    plt.tight_layout()
    plt.show()

