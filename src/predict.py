from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import IdentitySerializer


def predict_image(predictor, image_path, confidence=0.3, class_names=None):
    """Run inference on a JumpStart Object Detection endpoint for one image."""
    predictor.serializer = IdentitySerializer("application/x-image")
    predictor.deserializer = JSONDeserializer(accept="application/json;verbose")

    with open(image_path, "rb") as image_file:
        payload = image_file.read()

    result = predictor.predict(payload)

    # JumpStart OD returns normalized_boxes as [xmin, xmax, ymin, ymax].
    boxes = result["normalized_boxes"]
    classes = result["classes"]
    scores = result["scores"]
    labels = result.get("labels", classes)

    detections = []
    for box, class_id, score, label in zip(boxes, classes, scores, labels):
        if float(score) < confidence:
            continue

        xmin, xmax, ymin, ymax = box
        name = class_names[int(class_id)] if class_names else label
        detections.append(
            {
                "class": name,
                "class_id": int(class_id),
                "confidence": float(score),
                "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            }
        )

    return detections
