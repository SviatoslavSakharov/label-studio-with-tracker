from label_studio_sdk import Client
from pathlib import Path
import helper as hp
from utils.task_list import TaskList
from utils.trackers import TrackerNano, TrackerTamos
import hydra
from omegaconf import DictConfig, OmegaConf
import time


@hydra.main(config_path="", config_name="config")
def main(cfg: DictConfig) -> None:

    client = Client(url=cfg.url, api_key=cfg.api)
    if not hp.check_connection(client):
        print("Connection error")
        exit(1)

    project = client.get_project(cfg.project_id)

    print(f"Loading all tasks from project {cfg.project_id}")
    task_list = TaskList(project.get_tasks(), project, cfg.api, cfg.url)
    print(f"Loaded {len(task_list.get_all_tasks())} tasks")

    if cfg.tracker.start_task_id is not None:
        start_task = task_list.get_task_by_id(cfg.tracker.start_task_id)
    else:
        start_task = task_list.get_last_annotated_task()
    print(f"Start task id: {start_task.get_id()}")

    if start_task is not None:
        start_task.show_image_with_bboxes()
    else:
        raise ValueError(f"No task with id {cfg.start_task_id} found")

    if cfg.tracker.end_task_id is not None:
        n_frames_to_track = cfg.tracker.end_task_id - start_task.get_id()
    else:
        n_frames_to_track = cfg.tracker.n_frames_to_track

    print(f"Starting to track {n_frames_to_track} frames")
    if cfg.tracker.name == "nano":
        tracker = TrackerNano()
    elif cfg.tracker.name == "tamos":
        tracker = TrackerTamos()
    else:
        raise ValueError("Tracker type should be either 'nano' or 'tamos'")

    task_list.delete_present_annotations(start_task, n_frames_to_track)
    task_list.track_n_frames(start_task, n_frames_to_track, tracker)
    print("Uploading annotations")
    task_list.upload_annotations(start_task, n_frames_to_track)
    print(" Finished ")


if __name__ == "__main__":
    main()
