"""SageMaker PyTorch Inference Toolkit entry point.

Packaged into the model artifact (not baked into a custom image) alongside
requirements.txt -- the stock managed PyTorch inference container installs
requirements.txt and dispatches /ping and /invocations to the four functions below.
"""
import json
from pathlib import Path

from ultralytics import YOLO


def model_fn(model_dir):
    return YOLO(str(Path(model_dir) / "best.pt"))


def input_fn(request_body, content_type):
    if content_type != "application/x-image":
        raise ValueError(f"Unsupported content type: {content_type}")

    import io

    from PIL import Image

    return Image.open(io.BytesIO(request_body)).convert("RGB")


def predict_fn(input_image, model):
    return model.predict(input_image, verbose=False)[0]


def output_fn(result, accept):
    height, width = result.orig_shape
    detections = [
        {
            "class_id": int(box.cls[0]),
            "class": result.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "box": {
                "xmin": box.xyxy[0][0].item() / width,
                "ymin": box.xyxy[0][1].item() / height,
                "xmax": box.xyxy[0][2].item() / width,
                "ymax": box.xyxy[0][3].item() / height,
            },
        }
        for box in result.boxes
    ]
    return json.dumps({"detections": detections}), accept
