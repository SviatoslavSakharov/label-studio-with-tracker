import cv2
import sys
import numpy as np
import sys
from collections import OrderedDict

sys.path.append("C:\\Projects\\video_labeling\\label-studio-with-tracker\\pytracking")
from pytracking.evaluation import Tracker

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

if __name__ == "__main__":
    # Set up tracker.
    # Instead of MIL, you can also use

    # params = cv2.TrackerNano_Params()
    # params.backbone = "C:\\Projects\\drone\\models\\nanotrack_backbone_sim.onnx"
    # params.neckhead = "C:\\Projects\\drone\\models\\nanotrack_head_sim.onnx"
    # tracker = cv2.TrackerNano_create(params)
    tracker = Tracker("tamos", parameter_name="tamos_swin_base")
    params = tracker.get_parameters()
    print(f"params: {params}")
    tracker = tracker.create_tracker(params)
    print(f"tracker created: {tracker}")

    # Read video
    video = cv2.VideoCapture("C:\\Projects\\drone\\data\\videos\\fpv1.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)
    bbox2 = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    # ok = tracker.init(frame, bbox)
    next_object_id = 1
    out = tracker.initialize(
        frame,
        {
            "init_bbox": OrderedDict({1: bbox, 2: bbox2}),
            "init_object_ids": [
                1,
                2,
            ],
            "object_ids": [
                1,
                2,
            ],
            "sequence_object_ids": [1, 2],
        },
    )
    print(f"Tracker initialized")
    prev_output = OrderedDict(out)

    info = OrderedDict()

    fps_arr = []
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        info["previous_output"] = prev_output

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        # ok, bbox = tracker.update(frame)
        out = tracker.track(frame, info)
        prev_output = OrderedDict(out)
        bboxes = [bbox for _, bbox in out["target_bbox"].items()]

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        fps_arr.append(fps)
        fps_mean = np.mean(fps_arr)

        # Draw bounding box
        if ok:
            for bbox in bboxes:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(
            frame, f"FPS : {fps:.1f} Mean {fps_mean:.1f}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2
        )

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
