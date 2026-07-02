import json
import os
import time

import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = os.environ.get("MODEL_PATH", "fresh_vs_rotten_model.keras")
CLASS_NAMES = os.environ.get("CLASS_NAMES", "fresh,rotten").split(",")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))
FRAME_DELAY_SECONDS = float(os.environ.get("FRAME_DELAY_SECONDS", "0.2"))


model = None


def load_model():
    global model

    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    return model


def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype(np.float32)
    return np.expand_dims(image, axis=0)


def predict_frame(model, frame):
    image = preprocess_frame(frame)
    predictions = model.predict(image, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    label = CLASS_NAMES[class_index]

    return {
        "label": label,
        "confidence": confidence,
        "is_rotten": "rotten" in label.lower(),
    }


def handle_detection(result):
    if result["confidence"] < CONFIDENCE_THRESHOLD:
        return

    message = {
        "label": result["label"],
        "confidence": round(result["confidence"], 4),
        "is_rotten": result["is_rotten"],
        "timestamp": time.time(),
    }

    print(json.dumps(message))


def greengrass_infinite_infer_run():
    inference_model = load_model()
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Could not open Raspberry Pi camera.")

    try:
        while True:
            ret, frame = camera.read()

            if ret:
                result = predict_frame(inference_model, frame)
                handle_detection(result)

            time.sleep(FRAME_DELAY_SECONDS)
    finally:
        camera.release()


def function_handler(event, context):
    greengrass_infinite_infer_run()
    return
