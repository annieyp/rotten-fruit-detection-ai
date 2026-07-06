from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class DetectionModelNotConfiguredError(NotImplementedError):
    pass


class TensorFlowSavedModelDetector:
    def __init__(self, model_path, class_names=None):
        self.model_path = Path(model_path)
        self.model = tf.saved_model.load(str(self.model_path))
        self.detect_fn = self.model.signatures["serving_default"]
        self.class_names = class_names or []

    def class_name_for(self, class_id):
        class_index = int(class_id) - 1

        if 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]

        return f"class_{int(class_id)}"

    def predict(self, image, confidence=0.25):
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not read image: {image}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)[tf.newaxis, ...]
        outputs = self.detect_fn(input_tensor)

        boxes = outputs["detection_boxes"][0].numpy()
        scores = outputs["detection_scores"][0].numpy()
        classes = outputs["detection_classes"][0].numpy().astype(np.int64)
        height, width = rgb_image.shape[:2]

        detections = []

        for box, score, class_id in zip(boxes, scores, classes):
            if float(score) < confidence:
                continue

            ymin, xmin, ymax, xmax = box
            class_name = self.class_name_for(class_id)

            detections.append(
                {
                    "class": class_name,
                    "class_id": int(class_id),
                    "confidence": float(score),
                    "box": {
                        "xmin": float(xmin * width),
                        "ymin": float(ymin * height),
                        "xmax": float(xmax * width),
                        "ymax": float(ymax * height),
                    },
                }
            )

        return detections


def read_class_names(class_names_path):
    if class_names_path is None:
        return []

    return [
        line.strip()
        for line in Path(class_names_path).read_text().splitlines()
        if line.strip()
    ]


def build_model(*args, **kwargs):
    raise DetectionModelNotConfiguredError(
        "No object detection model has been selected yet. "
        "Choose a detector/framework first, then implement build_model for it."
    )


def load_model(
    model_path,
    model_type="tensorflow_saved_model",
    class_names_path=None,
    class_names=None,
):
    if model_type == "tensorflow_saved_model":
        return TensorFlowSavedModelDetector(
            model_path=model_path,
            class_names=class_names or read_class_names(class_names_path),
        )

    raise DetectionModelNotConfiguredError(
        f"Unsupported model_type '{model_type}'. Add an adapter in src/model.py."
    )
