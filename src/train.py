from sagemaker import hyperparameters as hp
from sagemaker.jumpstart.estimator import JumpStartEstimator


# SageMaker JumpStart pre-built EfficientDet D1 (SSD, 640x640, COCO17). AWS ships the
# training container with protobuf/pycocotools/TF/object_detection already resolved, so
# nothing runs the Object Detection API locally — the notebook only orchestrates.
DEFAULT_MODEL_ID = "tensorflow-od1-ssd-efficientdet-d1-640x640-coco17-tpu-8"
DEFAULT_MODEL_VERSION = "*"


def train(
    training_s3_uri,
    role,
    output_path=None,
    model_id=DEFAULT_MODEL_ID,
    model_version=DEFAULT_MODEL_VERSION,
    instance_type="ml.g5.xlarge",
    epochs=5,
    extra_hyperparameters=None,
):
    """Launch a JumpStart fine-tuning job on an ephemeral GPU instance."""
    hyperparameters = hp.retrieve_default(model_id=model_id, model_version=model_version)
    hyperparameters["epochs"] = str(epochs)
    if extra_hyperparameters:
        hyperparameters.update({k: str(v) for k, v in extra_hyperparameters.items()})

    estimator = JumpStartEstimator(
        model_id=model_id,
        model_version=model_version,
        role=role,
        instance_type=instance_type,
        hyperparameters=hyperparameters,
        output_path=output_path,
    )
    estimator.fit({"training": training_s3_uri})
    return estimator


def deploy(estimator, instance_type="ml.m5.xlarge"):
    """Deploy the fine-tuned model to a real-time endpoint. Remember to delete it after."""
    return estimator.deploy(instance_type=instance_type)
