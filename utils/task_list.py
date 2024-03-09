from utils.task import Task
from utils.trackers import create_tracker_nano
from label_studio_sdk import Project
from typing import List


class TaskList:
    def __init__(self, tasks_json: list, project: Project):
        self.project: Project = project
        self.tasks: List[Task] = [Task(task, project) for task in tasks_json]
        self.task_lookup: dict = {task.get_id(): task for task in self.tasks}

    def get_all_tasks(self):
        """Get all tasks"""
        return self.tasks

    def get_task_by_id(self, task_id: int):
        """Get a task by id"""
        return self.task_lookup[task_id]

    def get_last_annotated_task(self):
        """Get the last annotated task"""
        last_annotated_task = None
        for task in self.tasks[::-1]:
            if task.is_annotated():
                last_annotated_task = task
                break
        return last_annotated_task

    def track_n_frames(self, last_annotated_task, n_frames):
        frame = last_annotated_task.get_cv2_image()
        init_bboxes = last_annotated_task.get_annotations()["bboxes"]
        labels = last_annotated_task.get_annotations()["labels"]
        trackers = [create_tracker_nano() for _ in range(len(init_bboxes))]
        for i, tracker in enumerate(trackers):
            tracker.init(frame, init_bboxes[i])
        # ### get the next n_frames
        for task_id in range(last_annotated_task.get_id() + 1, last_annotated_task.get_id() + n_frames + 1):
            task = self.get_task_by_id(task_id)
            frame = task.get_cv2_image()
            for label, tracker in zip(labels, trackers):
                ok, bbox = tracker.update(frame)
                if ok:
                    task.set_annotation(label, bbox)
                else:
                    print(f"Tracker for label {label} for task {tasl.get_id()} failed")

    def upload_annotations(self, last_annotated_task, n_frames):
        for task_id in range(last_annotated_task.get_id() + 1, last_annotated_task.get_id() + n_frames + 1):
            task = self.get_task_by_id(task_id)
            task.upload_annotations()
