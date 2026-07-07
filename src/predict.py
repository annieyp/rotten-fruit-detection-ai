import keras
import keras_hub  # noqa: F401  (registers RetinaNet classes for load_model)
import tensorflow as tf

from model import load_model


def predict_image(model_path, image_path, image_size=640, confidence=0.3):
    model = load_model(model_path)

    image = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
    image = tf.image.resize_with_pad(image, image_size, image_size)
    image = tf.cast(image, tf.float32)

    outputs = model.predict(image[tf.newaxis, ...])

    boxes = outputs["boxes"][0]
    labels = outputs["labels"][0]
    scores = outputs["confidence"][0]

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if label < 0 or float(score) < confidence:
            continue

        ymin, xmin, ymax, xmax = [float(v) for v in box]
        detections.append(
            {
                "class_id": int(label),
                "confidence": float(score),
                "box": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
            }
        )

    return detections
