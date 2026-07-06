import cv2
import numpy as np
import tensorflow as tf

from dataset import CLASS_NAMES
from model import load_model


def predict_image(export_dir, image_path, confidence=0.3):
    detect_fn = load_model(export_dir)

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
        detections.append(
            {
                "class": CLASS_NAMES[class_id - 1] if 1 <= class_id <= len(CLASS_NAMES) else str(class_id),
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
