import cv2
from label_studio_sdk import Project


class Task:
    def __init__(
        self,
        task_json: dict,
        project: Project,
    ):
        self.json: dict = task_json
        self.project: Project = project
        self.id: int = task_json["id"]
        self.file_path: str = self._extract_file_path()
        self.annotations: dict = {"labels": [], "bboxes": []}
        self.image_shape: tuple = (None, None)
        self._extract_annotations()

    def get_id(self) -> int:
        """Get the task id"""
        return self.id

    def get_file_path(self) -> str:
        """Get the file path"""
        return self.file_path

    def get_annotations(self) -> dict:
        """Get the annotations"""
        return self.annotations

    def is_annotated(self) -> bool:
        """Check if the task is annotated"""
        return len(self.annotations["labels"]) > 0

    def get_json(self):
        return self.json

    def get_cv2_image(self):
        """Get the image as a cv2 image"""
        image = cv2.imread(str(self.file_path))
        if self.image_shape == (None, None):
            self.image_shape = image.shape[:2]
        return image

    def clear_annotations(self):
        self.annotations = {"labels": [], "bboxes": []}

    def set_annotation(self, label, bbox):
        self.annotations["labels"].append(label)
        self.annotations["bboxes"].append(bbox)

    def show_image_with_bboxes(self):
        img_path = self.file_path
        print("Showing image: ", img_path)
        image = cv2.imread(str(img_path))
        annotations = self.get_annotations()
        bboxes = annotations["bboxes"]
        labels = annotations["labels"]
        for i, bbox in enumerate(bboxes):
            x, y, w, h = [int(coord) for coord in bbox]
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

    def upload_annotations(self):
        results = self._generate_results()
        self.project.create_annotation(self.id, result=results)

    def _extract_file_path(self):
        if self.json["storage_filename"] is not None:
            return self.json["storage_filename"]

        raise ValueError(f"Tasks id {self.id} no file name found")

    def _extract_annotations(self):
        if len(self.json["annotations"]) > 0:
            annotations = self.json["annotations"][0]["result"]
            try:
                image_shape = (annotations[0]["original_height"], annotations[0]["original_width"])
                self.image_shape = image_shape
                for annotation in annotations:
                    x = annotation["value"]["x"]
                    y = annotation["value"]["y"]
                    width = annotation["value"]["width"]
                    height = annotation["value"]["height"]
                    bbox = [
                        x / 100,  # if x > 1 else x,
                        y / 100,  # if y > 1 else y,
                        width / 100,  # if width > 1 else width,
                        height / 100,  # if height > 1 else height,
                    ]
                    bbox_cv2 = self._convert_bbox_from_yolo_to_cv2(bbox)
                    label = annotation["value"]["rectanglelabels"][0]
                    self.annotations["labels"].append(label)
                    self.annotations["bboxes"].append(bbox_cv2)
            except:
                print(f"WARNING: Task {self.id} cannot extract annotations")

    def _convert_bbox_from_yolo_to_cv2(self, bbox):
        x = bbox[0] * self.image_shape[1]
        y = bbox[1] * self.image_shape[0]
        w = bbox[2] * self.image_shape[1]
        h = bbox[3] * self.image_shape[0]
        return (x, y, w, h)

    def _convert_bbox_from_cv2_to_yolo(self, bbox):
        x, y, w, h = bbox
        return (x / self.image_shape[1], y / self.image_shape[0], w / self.image_shape[1], h / self.image_shape[0])

    def _generate_results(self):
        """Generate results annotations for the task"""
        results = []
        for bbox, label in zip(self.annotations["bboxes"], self.annotations["labels"]):
            bbox = self._convert_bbox_from_cv2_to_yolo(bbox)
            results.append(
                {
                    "original_width": self.image_shape[1],
                    "original_height": self.image_shape[0],
                    "image_rotation": 0,
                    "value": {
                        "x": bbox[0] * 100,
                        "y": bbox[1] * 100,
                        "width": bbox[2] * 100,
                        "height": bbox[3] * 100,
                        "rotation": 0,
                        "rectanglelabels": [label],
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "origin": "manual",
                }
            )
        return results
