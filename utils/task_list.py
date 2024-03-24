from utils.task import Task
from label_studio_sdk import Project
from typing import List
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor
import time


class TaskList:
    def __init__(self, tasks_json: list, project: Project, api_key: str, url: str):
        self.project: Project = project
        self.tasks: List[Task] = [Task(task, project) for task in tasks_json]
        self.task_lookup: dict = {task.get_id(): task for task in self.tasks}
        self.api_key = api_key
        self.header = {"Authorization": f"Token {self.api_key}"}
        self.url = url

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

    def track_n_frames(self, starting_task, n_frames, tracker):
        frame = starting_task.get_cv2_image()
        init_bboxes = starting_task.get_annotations()["bboxes"]
        labels = starting_task.get_annotations()["labels"]
        tracker.initialize(frame, init_bboxes, labels)
        print(
            f"Tracking {n_frames} frames from task {starting_task.get_id()} to task {starting_task.get_id() + n_frames}"
        )
        for task_id in tqdm(range(starting_task.get_id() + 1, starting_task.get_id() + n_frames + 1)):
            task = self.get_task_by_id(task_id)
            frame = task.get_cv2_image()
            labels, bboxes = tracker.track(frame)
            for label, bbox in zip(labels, bboxes):
                task.set_annotation(label, bbox)

    def upload_annotations(self, starting_task, n_frames):
        for task_id in range(starting_task.get_id() + 1, starting_task.get_id() + n_frames + 1):
            task = self.get_task_by_id(task_id)
            task.upload_annotations()

    def delete_present_annotations(self, starting_task, n_frames):
        annotation_ids = []
        for task_id in range(starting_task.get_id() + 1, starting_task.get_id() + n_frames + 1):
            task = self.get_task_by_id(task_id)
            for annotation in task.get_json()["annotations"]:
                annotation_ids.append(annotation["id"])
            task.clear_annotations()
        print(
            f"Deleting {len(annotation_ids)} annotations from tasks {starting_task.get_id() + 1} to {starting_task.get_id() + n_frames}"
        )
        self.delete_annotations_request(annotation_ids)

    def delete_annotations_request(self, annotation_ids):
        def delete_annotation(annotation_id):
            url = f"{self.url}/api/annotations/{annotation_id}"
            requests.delete(url, headers=self.header)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(delete_annotation, annotation_ids), total=len(annotation_ids)))
