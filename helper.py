import cv2
from pathlib import Path
from tqdm import tqdm


def generate_results(image_shape, bboxes, labels):
    ## bboxes in yolo format
    results = []
    for i, bbox in enumerate(bboxes):
        results.append(
            {
                "original_width": image_shape[1],
                "original_height": image_shape[0],
                "image_rotation": 0,
                "value": {
                    "x": bbox[0] * 100,
                    "y": bbox[1] * 100,
                    "width": bbox[2] * 100,
                    "height": bbox[3] * 100,
                    "rotation": 0,
                    "rectanglelabels": [labels[i]],
                },
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "origin": "manual",
            }
        )
    return results


def get_annotations(task):
    annotation_results = task["annotations"][0]["result"]
    file_name = task["file_upload"]
    bboxes = []
    labels = []
    for annotation in annotation_results:
        x = annotation["value"]["x"]
        y = annotation["value"]["y"]
        width = annotation["value"]["width"]
        height = annotation["value"]["height"]
        bboxes.append(
            [
                x / 100 if x > 1 else x,
                y / 100 if y > 1 else y,
                width / 100 if width > 1 else width,
                height / 100 if height > 1 else height,
            ]
        )
        labels.append(annotation["value"]["rectanglelabels"][0])
    return {"file_name": file_name, "bboxes": bboxes, "labels": labels}


def get_last_annotations(tasks):
    last_annotated_task_id = None
    for task in tasks[::-1]:
        if task["is_labeled"] == True:
            last_annotated_task_id = task["id"]
            annotations = get_annotations(task)
            break

    if last_annotated_task_id is None:
        print("No labeled tasks found")
        return None
    else:
        return annotations | {"task_id": last_annotated_task_id}


def check_connection(client):
    out = client.check_connection()
    if out["status"] == "UP":
        print("Connection to Label Studio is successful")
        return True
    else:
        print("Connection to Label Studio failed")
        return False


def show_image_with_bboxes(info, local_folder):
    img_name = info["file_name"].split("-")[-1]
    print("Image name: ", img_name)
    image = cv2.imread(str(local_folder / img_name))
    bboxes = info["bboxes"]
    labels = info["labels"]
    for i, bbox in enumerate(bboxes):
        #### bbxox in yolo format
        x = int(bbox[0] * image.shape[1])
        y = int(bbox[1] * image.shape[0])
        w = int(bbox[2] * image.shape[1])
        h = int(bbox[3] * image.shape[0])
        print("x, y, w, h: ", x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            labels[i],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_tracker_nano():
    params = cv2.TrackerNano_Params()
    params.backbone = "C:\\Projects\\drone\\models\\nanotrack_backbone_sim.onnx"
    params.neckhead = "C:\\Projects\\drone\\models\\nanotrack_head_sim.onnx"
    tracker = cv2.TrackerNano_create(params)
    return tracker


def convert_bbox_from_yolo_to_cv2(bbox, image_shape):
    x = int(bbox[0] * image_shape[1])
    y = int(bbox[1] * image_shape[0])
    w = int(bbox[2] * image_shape[1])
    h = int(bbox[3] * image_shape[0])
    return (x, y, w, h)


def convert_bbox_from_cv2_to_yolo(bbox, image_shape):
    x, y, w, h = bbox
    return (x / image_shape[1], y / image_shape[0], w / image_shape[1], h / image_shape[0])


def track_n_frames(tasks, last_annotations, n_frames, local_folder):
    last_id = last_annotations["task_id"]
    frame = cv2.imread(str(local_folder / last_annotations["file_name"].split("-")[-1]))
    init_bboxes = [convert_bbox_from_yolo_to_cv2(bbox, frame.shape) for bbox in last_annotations["bboxes"]]
    labels = last_annotations["labels"]
    trackers = [create_tracker_nano() for _ in range(len(init_bboxes))]
    for i, tracker in enumerate(trackers):
        print("Init bbox: ", init_bboxes[i])
        tracker.init(frame, init_bboxes[i])
    annotations = []
    ### get the next n_frames
    for task in tasks:
        if task["id"] > last_id and task["id"] <= last_id + n_frames:
            frame = cv2.imread(str(local_folder / task["file_upload"].split("-")[-1]))
            bboxes = []
            for i, tracker in enumerate(trackers):
                ok, bbox = tracker.update(frame)
                if ok:
                    bboxes.append(convert_bbox_from_cv2_to_yolo(bbox, frame.shape))
                else:
                    print(f"Tracker {i} failed")
            annotations.append(
                {
                    "file_name": task["file_upload"],
                    "bboxes": bboxes,
                    "labels": labels,
                    "task_id": task["id"],
                    "image_shape": frame.shape,
                }
            )
        elif task["id"] > last_id + n_frames:
            break
    return annotations


def upload_annotations(project, annotations):
    for annotation in tqdm(annotations):
        results = generate_results(annotation["image_shape"], annotation["bboxes"], annotation["labels"])
        project.create_annotation(annotation["task_id"], result=results)
