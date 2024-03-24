from tqdm import tqdm
import requests

# from multiprocessing import Pool
from tqdm import tqdm
from utils.task import Task
from typing import List
from concurrent.futures import ThreadPoolExecutor


def check_connection(client):
    out = client.check_connection()
    if out["status"] == "UP":
        print("Connection to Label Studio is successful")
        return True
    else:
        print("Connection to Label Studio failed")
        return False


def delete_annotations(header, annotation_ids):
    # use muotiprocessing to delete annotations
    def delete_annotation(annotation_id):
        url = f"http://localhost:8080/api/annotations/{annotation_id}"
        requests.delete(url, headers=header)

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Wrapping executor.map in tqdm allows you to display a progress bar
        list(tqdm(executor.map(delete_annotation, annotation_ids), total=len(annotation_ids)))
    # for annotation_id in tqdm(annotation_ids):
    #     delete_annotation(annotation_id)


def get_annotation_ids(all_tasks, task_ids_to_delete):
    annotation_ids = []
    for task in all_tasks:
        if task["id"] in task_ids_to_delete:
            for annotation in task["annotations"]:
                annotation_ids.append(annotation["id"])
    return annotation_ids


def delete_annotations_from_tasks(header, all_tasks, task_ids_to_delete):
    annotation_ids = get_annotation_ids(all_tasks, task_ids_to_delete)
    print(f"Deleting {annotation_ids} annotations")
    # delete_annotations(header, annotation_ids)
