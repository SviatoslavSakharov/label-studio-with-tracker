import cv2

import sys
import numpy as np

sys.path.append("C:\\Projects\\video_labeling\\label-studio-with-tracker\\pytracking")
sys.path.append("C:\\Projects\\video_labeling\\label-studio-with-tracker\\utils")
from pytracking.evaluation import Tracker
from collections import OrderedDict
from omegaconf import DictConfig
from task_list import TaskList


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


class Interpolator:
    def __init__(self, task_list: TaskList, cfg: DictConfig):
        self.start_task_id = cfg.tracker.start_task_id
        self.end_task_id = cfg.tracker.end_task_id
        self.intermediate_task_ids = cfg.tracker.intermediate_task_ids
        self.intermediate_task_ids.sort()
        self.check_intermediate_task_ids()

        self.start_annotations = task_list.get_task_by_id(self.start_task_id).get_annotations()
        self.end_annotations = task_list.get_task_by_id(self.end_task_id).get_annotations()
        self.intermediate_tasks_annotations = [
            task_list.get_task_by_id(task_id).get_annotations() for task_id in self.intermediate_task_ids
        ]
        self.check_for_annotations()
        self.interpolation_info = {}
        self.initialize()

    def check_for_annotations(self):
        if len(self.start_annotations["labels"]) == 0:
            raise ValueError("Start task has no annotations")
        if len(self.end_annotations["labels"]) == 0:
            raise ValueError("End task has no annotations")
        for i, intermediate_task_annotations in enumerate(self.intermediate_tasks_annotations):
            if len(intermediate_task_annotations["labels"]) == 0:
                raise ValueError(f"Intermediate task {i} has no annotations")

    def check_intermediate_task_ids(self):
        for task_id in self.intermediate_task_ids:
            if task_id < self.start_task_id or task_id > self.end_task_id:
                raise ValueError(f"Intermediate task id {task_id} is not in the range of start and end task ids")

    def calculate_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (w1 + 1) * (h1 + 1)
        boxBArea = (w2 + 1) * (h2 + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _sort_labels(self, start_annotations, end_annotations, start_id, end_id):
        new_end_annotations = {"labels": [], "bboxes": []}
        start_bboxes = np.array(start_annotations["bboxes"])
        end_bboxes = np.array(end_annotations["bboxes"])
        for i, start_bbox in enumerate(start_bboxes):
            # distances = np.linalg.norm(start_bbox - end_bboxes, axis=1)
            distances = [self.calculate_iou(start_bbox, end_bbox) for end_bbox in end_bboxes]
            closest_idx = np.argmax(distances)
            if np.all(np.array(distances) == 0):
                print(f"WARNING: No intersection between start and end bboxes for ids {start_id} and {end_id}")
                distances = np.linalg.norm(start_bbox - end_bboxes, axis=1)
                closest_idx = np.argmin(distances)
            new_end_annotations["labels"].append(end_annotations["labels"][closest_idx])
            new_end_annotations["bboxes"].append(end_annotations["bboxes"][closest_idx])
            assert (
                start_annotations["labels"][i] == end_annotations["labels"][closest_idx]
            ), f"Labels should be the same start_id {start_id} end_id {end_id}, distance {distances}, start_labels {start_annotations['labels']}, end_labels {end_annotations['labels']}"
        assert len(start_annotations) == len(new_end_annotations), "Number of labels should be the same"
        return start_annotations, new_end_annotations

    def initialize(self):
        all_ids = [self.start_task_id] + self.intermediate_task_ids + [self.end_task_id]
        all_annotations = [self.start_annotations] + self.intermediate_tasks_annotations + [self.end_annotations]
        for i in range(len(all_ids) - 1):
            start_annotations, end_annotations = self._sort_labels(
                all_annotations[i], all_annotations[i + 1], all_ids[i], all_ids[i + 1]
            )
            start_id, end_id = all_ids[i], all_ids[i + 1]
            labels = start_annotations["labels"]
            velocities = []
            for label_id in range(len(labels)):
                start_bbox = np.array(start_annotations["bboxes"][label_id])
                end_bbox = np.array(end_annotations["bboxes"][label_id])
                velocity = (end_bbox - start_bbox) / (end_id - start_id)
                velocities.append(velocity)
            self.interpolation_info[(start_id, end_id)] = {
                "labels": labels,
                "velocities": velocities,
                "start_bboxes": start_annotations["bboxes"],
                "end_bboxes": end_bbox,
                "end_labels": end_annotations["labels"],
            }

    def track(self, task_id):
        for (start_id, end_id), info in self.interpolation_info.items():
            if task_id >= start_id and task_id <= end_id:
                labels = info["labels"]
                velocities = info["velocities"]
                start_bboxes = info["start_bboxes"]
                current_frame = task_id - start_id
                bboxes = [
                    start_bbox + velocity * current_frame for velocity, start_bbox in zip(velocities, start_bboxes)
                ]
                return labels, bboxes
        raise ValueError(f"Task id {task_id} is not in the range of interpolation")
