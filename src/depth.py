import sys
import os
import torch
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "third_party")
)
from depth_anything_v2.dpt import DepthAnythingV2

def load_depth_model():
    """
    Load and initialize the metric Depth Anything V2 model.

    The model is configured for outdoor metric depth estimation
    with a maximum depth of 80 meters.

    Returns:
        Initialized Depth Anything V2 model in evaluation mode.
    """
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}

    max_depth = 80

    depth_model = DepthAnythingV2(**{**model_config, 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load('models/depth_anything_v2_metric_vkitti_vits.pth', map_location='cpu'))
    depth_model = depth_model.to(device)
    depth_model.eval()
    return depth_model

def depth_map(frame, depth_model):
    """
    Estimate a metric depth map for an RGB frame.

    Args:
        frame: Input RGB image frame.
        depth_model: Initialized Depth Anything V2 model.

    Returns:
        Depth map containing estimated depth in meters for each pixel.
    """
    depth = depth_model.infer_image(frame)
    return depth