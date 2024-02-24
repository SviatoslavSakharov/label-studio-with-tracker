from label_studio_sdk import Client
import cv2
from pathlib import Path
import helper as hp

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "a6021e2d67c73b759f4a27967a4932a3fd0eb6df"
PROJECT_ID = "5"
LOCAL_FOLDER = Path("fpv1")


if __name__ == "__main__":

    client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    if not hp.check_connection(client):
        exit(1)
    project = client.get_project(PROJECT_ID)
    tasks = project.get_tasks()
    last_annotations = hp.get_last_annotations(tasks)
    print(f"Task id: {last_annotations['task_id']}")
    # if last_annotations is not None:
    #     hp.show_image_with_bboxes(last_annotations, LOCAL_FOLDER)
    if last_annotations is None:
        print("No labeled tasks found")
        exit(1)
    print(f"Last annotations: {last_annotations}")
    tracked_annotations = hp.track_n_frames(tasks, last_annotations, 25, LOCAL_FOLDER)
    hp.upload_annotations(project, tracked_annotations)
