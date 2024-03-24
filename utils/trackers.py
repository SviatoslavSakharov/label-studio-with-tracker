import cv2

import sys

sys.path.append("C:\\Projects\\video_labeling\\label-studio-with-tracker\\pytracking")
from pytracking.evaluation import Tracker
from collections import OrderedDict


class TrackerNano:
    def __init__(
        self,
    ):
        self.params = cv2.TrackerNano_Params()
        self.params.backbone = "C:\\Projects\\drone\\models\\nanotrack_backbone_sim.onnx"
        self.params.neckhead = "C:\\Projects\\drone\\models\\nanotrack_head_sim.onnx"
        self.trackers = []
        self.labels = []

    def create_tracker(self):
        tracker = cv2.TrackerNano_create(self.params)
        return tracker

    def initialize(self, frame, init_bboxes, labels):
        self.trackers = [self.create_tracker() for _ in range(len(init_bboxes))]
        self.labels = labels
        assert len(self.trackers) == len(self.labels), "Number of trackers and labels should be the same"
        for i, tracker in enumerate(self.trackers):
            tracker.init(frame, init_bboxes[i])

    def track(self, frame):
        out_labels = []
        out_bboxes = []
        for label, tracker in zip(self.labels, self.trackers):
            ok, bbox = tracker.update(frame)
            if ok:
                out_labels.append(label)
                out_bboxes.append(bbox)
            else:
                print(f"Tracker for label {label} failed")
        return out_labels, out_bboxes


class TrackerTamos:
    def __init__(self, parameter_name="tamos_swin_base"):
        tracker = Tracker("tamos", parameter_name=parameter_name)
        params = tracker.get_parameters()
        self.tracker = tracker.create_tracker(params)
        self.info = {}
        self.labels = []

    def initialize(self, frame, init_bboxes, labels):
        obj_bboxes = {i: bbox for i, bbox in enumerate(init_bboxes)}
        self.labels = labels
        out = self.tracker.initialize(
            frame,
            {
                "init_bbox": obj_bboxes,
                "init_object_ids": obj_bboxes.keys(),
                "object_ids": obj_bboxes.keys(),
                "sequence_object_ids": obj_bboxes.keys(),
            },
        )
        # print("out: ", out)
        self.info["prev_output"] = out
        print(f"Tamos tracker is initialized")

    def track(self, frame):
        out = self.tracker.track(frame, self.info)
        # print("out: ", out)
        self.info["prev_output"] = out
        out_bboxes = [bbox for _, bbox in out["target_bbox"].items()]
        out_labels = self.labels
        return out_labels, out_bboxes
