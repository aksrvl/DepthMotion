from ultralytics import YOLO

def load_tracking_model():
    """
    Load the YOLO model used for object detection and tracking.

    Returns:
        Initialized YOLO model.
    """
    model = YOLO("yolo26n.pt") 
    return model

def track_objects(model, frame):
    """
    Detect and track objects in a video frame using ByteTrack.

    Args:
        model: Initialized YOLO model.
        frame: Current input video frame.

    Returns:
        List of (track_id, bounding_box) tuples for tracked objects.
        Each bounding box contains (x1, y1, x2, y2) pixel coordinates.
    """
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)
    tracked_objects = []
    for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            if result.boxes.id is None:
                continue
            track_ids = result.boxes.id.cpu().numpy()
            for box, track_id in zip(boxes, track_ids):
                tracked_objects.append((track_id, box))

    return tracked_objects
        