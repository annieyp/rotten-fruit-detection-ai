import io, json
from pathlib import Path
from flask import Flask, Response, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(Path("/opt/ml/model") / "best.pt"))
    return _model

@app.route("/ping", methods=["GET"])
def ping():
    get_model()
    return Response(status=200)

@app.route("/invocations", methods=["POST"])
def invocations():
    image = Image.open(io.BytesIO(request.data)).convert("RGB")
    result = get_model().predict(image, verbose=False)[0]
    h, w = result.orig_shape
    detections = [
        {
            "class_id": int(box.cls[0]),
            "class": result.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "box": {
                "xmin": box.xyxy[0][0].item() / w, "ymin": box.xyxy[0][1].item() / h,
                "xmax": box.xyxy[0][2].item() / w, "ymax": box.xyxy[0][3].item() / h,
            },
        }
        for box in result.boxes
    ]
    return Response(json.dumps({"detections": detections}), mimetype="application/json")
