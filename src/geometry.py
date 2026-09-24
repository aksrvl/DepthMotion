import numpy as np

def load_intrinsics():
    """
    Load camera intrinsic parameters from the Virtual KITTI 2
    intrinsic calibration file.

    Returns:
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Principal point x-coordinate in pixels.
        cy: Principal point y-coordinate in pixels.
    """
    with open ('./data/intrinsic.txt', 'r') as file:
        lines = file.readlines()

    values = lines[1].split()
    
    fx = float(values[2])
    fy = float(values[3])
    cx = float(values[4])
    cy = float(values[5])

    return fx, fy, cx, cy

def bbox_to_bev(box, depth, fx, cx):
    """
    Estimate an object's position in Bird's-Eye View (BEV).

    The object's depth is estimated as the median depth within
    the central region of its bounding box. The horizontal BEV
    position is then computed using the pinhole camera model.

    Args:
        box: Bounding box coordinates (x1, y1, x2, y2) in pixels.
        depth: Metric depth map containing depth values in meters.
        fx: Horizontal focal length in pixels.
        cx: Principal point x-coordinate in pixels.

    Returns:
        X: Lateral position relative to the camera in meters.
        Z: Forward depth from the camera in meters.
    """
    x1, y1, x2, y2 = box
    
    x_1 = int(x1 + (x2 - x1)*0.3)
    x_2 = int(x2 - (x2 - x1)*0.3)
    y_1 = int(y1 + (y2 - y1)*0.3)
    y_2 = int(y2 - (y2 - y1)*0.3)
    
    centre_region = depth[y_1:y_2, x_1:x_2]
    pred_depth = np.median(centre_region)

    u = int((x1 +x2)/2)
    Z = pred_depth
    X = (u - cx)*(Z/fx)

    return X, Z
        