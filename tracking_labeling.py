from label_studio_sdk import Client
import cv2
from pathlib import Path
import helper as hp
import json
from utils.task_list import TaskList
import time

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "a6021e2d67c73b759f4a27967a4932a3fd0eb6df"
PROJECT_ID = "7"
LOCAL_FOLDER = Path("C:\\Projects\\video_labeling\\data\\0001-0010")
N_FRAMES_TO_TRACK = 5
END_TRACKING_TASK_ID = None  # 260


if __name__ == "__main__":

    client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    if not hp.check_connection(client):
        print("Connection error")
        exit(1)

    project = client.get_project(PROJECT_ID)

    task_list = TaskList(project.get_tasks(), project)

    last_annotated_task = task_list.get_last_annotated_task()
    print(f"Last annotated task id: {last_annotated_task.get_id()}")

    if last_annotated_task is not None:
        last_annotated_task.show_image_with_bboxes()
    else:
        raise ValueError("No labeled tasks found")

    if END_TRACKING_TASK_ID is not None:
        n_frames_to_track = END_TRACKING_TASK_ID - last_annotated_task.get_id()
    else:
        n_frames_to_track = N_FRAMES_TO_TRACK
    print(f"Starting to track {n_frames_to_track} frames")
    task_list.track_n_frames(last_annotated_task, n_frames_to_track)
    print(f" Finished ")
    print(f"Uploading annotations")
    task_list.upload_annotations(last_annotated_task, n_frames_to_track)
    print(f" Finished ")
