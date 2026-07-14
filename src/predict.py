from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import IdentitySerializer


def predict_image(predictor, image_path, confidence=0.3):
    """Run inference on a YOLOv8 SageMaker endpoint for one image.

    Expects your image's /invocations route to accept "application/x-image" (raw JPEG
    bytes) and return application/json shaped like:
        {"detections": [{"class": str, "class_id": int, "confidence": float,
                          "box": {"xmin", "ymin", "xmax", "ymax"}}, ...]}
    with box coordinates normalized to [0, 1].
    """
    predictor.serializer = IdentitySerializer("application/x-image")
    predictor.deserializer = JSONDeserializer(accept="application/json")

    with open(image_path, "rb") as image_file:
        payload = image_file.read()

    result = predictor.predict(payload)
    return [d for d in result["detections"] if d["confidence"] >= confidence]
