from pathlib import Path

import cv2
import tensorflow as tf

from model import read_label_map_names


def load_detector(export_dir):
    saved_model = tf.saved_model.load(str(Path(export_dir) / "saved_model"))
    return saved_model.signatures["serving_default"]


def predict_image(
    export_dir,
    image_path,
    label_map_path="annotations/label_map.pbtxt",
    confidence=0.3,
):
    detect_fn = load_detector(export_dir)
    class_names = read_label_map_names(label_map_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    input_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)[tf.newaxis, ...]

    outputs = detect_fn(input_tensor)
    boxes = outputs["detection_boxes"][0].numpy()
    scores = outputs["detection_scores"][0].numpy()
    classes = outputs["detection_classes"][0].numpy().astype(int)

    detections = []
    for box, score, class_id in zip(boxes, scores, classes):
        if score < confidence:
            continue

        ymin, xmin, ymax, xmax = box
        name = class_names[class_id - 1] if 1 <= class_id <= len(class_names) else str(class_id)
        detections.append(
            {
                "class": name,
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
