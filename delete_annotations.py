from helper import delete_annotations_from_tasks
from label_studio_sdk import Client
import numpy as np

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "a6021e2d67c73b759f4a27967a4932a3fd0eb6df"

if __name__ == "__main__":
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    print(ls.check_connection())
    project = ls.get_project(7)
    tasks = project.get_tasks()

    headers = {"Authorization": f"Token {API_KEY}"}
    task_ids_to_delete = np.arange(132, 137)
    delete_annotations_from_tasks(headers, tasks, task_ids_to_delete)
