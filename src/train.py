from sagemaker.estimator import Estimator


DEFAULT_YOLO_MODEL = "yolo26n.pt"


def train(
    training_s3_uri,
    role,
    image_uri,
    output_path=None,
    model_name=DEFAULT_YOLO_MODEL,
    instance_type="ml.g5.2xlarge",
    epochs=75,
    batch_size=16,
    learning_rate=0.001,
    optimizer="SGD",
    imgsz=640,
    extra_hyperparameters=None,
):
    """Fine-tune YOLO26 on an ephemeral GPU instance using a custom (bring-your-own)
    container image -- YOLO26 isn't in SageMaker JumpStart's built-in model zoo, so this
    runs against whatever image_uri you built and pushed to ECR yourself.

    training_s3_uri should point at a Roboflow "YOLOv8" (Ultralytics) format export
    uploaded as-is -- train/images, train/labels, valid/images, valid/labels, and
    data.yaml at the root. No local conversion step needed.

    Contract your image's training entrypoint must satisfy (standard SageMaker BYOC):
      - Runs `docker run <image> train`, i.e. a `train` executable on PATH.
      - Reads hyperparameters from /opt/ml/input/config/hyperparameters.json (values
        are strings: "model", "epochs", "batch", "lr0", "optimizer", "imgsz").
      - Reads the dataset from /opt/ml/input/data/training (the "training" channel
        below). Roboflow's data.yaml often has a `path` that's absolute or relative to
        wherever it was originally downloaded -- your entrypoint must rewrite `path` to
        the actual mounted channel directory before handing the yaml to a trainer, since
        that directory differs job to job.
      - Writes the final model artifact(s) under /opt/ml/model.
    """
    hyperparameters = {
        "model": model_name,
        "epochs": epochs,
        "batch": batch_size,
        "lr0": learning_rate,
        "optimizer": optimizer,
        "imgsz": imgsz,
    }
    if extra_hyperparameters:
        hyperparameters.update(extra_hyperparameters)
    hyperparameters = {k: str(v) for k, v in hyperparameters.items()}

    estimator = Estimator(
        role=role,
        image_uri=image_uri,
        instance_count=1,
        instance_type=instance_type,
        max_run=360000,
        hyperparameters=hyperparameters,
        output_path=output_path,
    )
    estimator.fit({"training": training_s3_uri})
    return estimator


def deploy(estimator, instance_type="ml.m5.xlarge"):
    """Deploy the fine-tuned model to a real-time endpoint, using the same image's
    `serve` entrypoint (must expose GET /ping and POST /invocations on port 8080, per
    SageMaker's BYOC serving contract -- see predict.predict_image for the expected
    /invocations request/response format). Remember to delete it after."""
    return estimator.deploy(instance_type=instance_type)
