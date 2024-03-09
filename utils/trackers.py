import cv2


def create_tracker_nano():
    params = cv2.TrackerNano_Params()
    params.backbone = "C:\\Projects\\drone\\models\\nanotrack_backbone_sim.onnx"
    params.neckhead = "C:\\Projects\\drone\\models\\nanotrack_head_sim.onnx"
    tracker = cv2.TrackerNano_create(params)
    return tracker
