import json
import os
import time

import cv2

from model import load_model


MODEL_PATH = os.environ.get("MODEL_PATH", "exported-model/saved_model")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "tensorflow_saved_model")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "artifacts/class_names.txt")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))
FRAME_DELAY_SECONDS = float(os.environ.get("FRAME_DELAY_SECONDS", "0.2"))


model = None


def get_model():
    global model

    if model is None:
        model = load_model(
            MODEL_PATH,
            model_type=MODEL_TYPE,
            class_names_path=CLASS_NAMES_PATH,
        )

    return model


def normalize_detection(detection):
    class_name = detection["class"]

    return {
        "class": class_name,
        "confidence": round(float(detection["confidence"]), 4),
        "is_rotten": class_name.lower().startswith(("bad", "rotten")),
        "box": detection["box"],
    }


def detect_frame(model, frame):
    detections = model.predict(frame, confidence=CONFIDENCE_THRESHOLD)
    return [normalize_detection(detection) for detection in detections]


def handle_detections(detections):
    if not detections:
        return

    print(
        json.dumps(
            {
                "timestamp": time.time(),
                "detections": detections,
            }
        )
    )


def greengrass_infinite_infer_run():
    inference_model = get_model()
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Could not open Raspberry Pi camera.")

    try:
        while True:
            ret, frame = camera.read()

            if ret:
                detections = detect_frame(inference_model, frame)
                handle_detections(detections)

            time.sleep(FRAME_DELAY_SECONDS)
    finally:
        camera.release()


def function_handler(event, context):
    greengrass_infinite_infer_run()
    return
