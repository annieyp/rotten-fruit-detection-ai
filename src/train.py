import tarfile
from pathlib import Path

import boto3
from sagemaker import hyperparameters as hp
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.estimator import Estimator


# SageMaker JumpStart pre-built EfficientDet D1 (SSD, 640x640, COCO17). AWS ships the
# training container with the TF Object Detection API already resolved, so nothing runs
# that API locally -- we only orchestrate the job.
DEFAULT_MODEL_ID = "tensorflow-od1-ssd-efficientdet-d1-640x640-coco17-tpu-8"
DEFAULT_MODEL_VERSION = "*"


def download_and_patch_script(model_id, model_version, source_dir="od_script"):
    """Download AWS's JumpStart OD training script and patch its retracing leak.

    The trainer's train step (od_script/train.py) is a @tf.function called with Python
    lists of variable-shape groundtruth box tensors ([N_i, 4] with N_i differing per
    image). tf.function keys its trace on input shapes, so it recompiles every step and
    never frees the old traces -> host RAM climbs until SIGKILL, on any instance size.
    reduce_retracing=True relaxes the varying dimension and stops the recompiles.

    Idempotent: reuses an already-downloaded/patched source_dir. Returns its path.
    """
    source_dir = Path(source_dir)

    if not source_dir.exists():
        script_uri = script_uris.retrieve(
            model_id=model_id, model_version=model_version, script_scope="training"
        )
        bucket, key = script_uri.replace("s3://", "").split("/", 1)
        archive = "od_sourcedir.tar.gz"
        boto3.client("s3").download_file(bucket, key, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(source_dir)

    train_py = source_dir / "train.py"
    src = train_py.read_text()
    if "reduce_retracing=True" not in src:
        if "@tf.function\n" not in src:
            raise RuntimeError(f"Unexpected layout in {train_py}; cannot patch retracing.")
        src = src.replace("@tf.function\n", "@tf.function(reduce_retracing=True)\n", 1)
        train_py.write_text(src)
        print(f"Patched {train_py} with reduce_retracing=True")
    else:
        print(f"{train_py} already patched")

    return str(source_dir)


def train(
    training_s3_uri,
    role,
    output_path=None,
    model_id=DEFAULT_MODEL_ID,
    model_version=DEFAULT_MODEL_VERSION,
    instance_type="ml.g5.2xlarge",
    epochs=5,
    extra_hyperparameters=None,
    source_dir="od_script",
):
    """Fine-tune on an ephemeral GPU instance using AWS's training script, patched to
    stop the tf.function retracing memory leak (see download_and_patch_script)."""
    hyperparameters = hp.retrieve_default(model_id=model_id, model_version=model_version)
    hyperparameters["epochs"] = str(epochs)
    if extra_hyperparameters:
        hyperparameters.update({k: str(v) for k, v in extra_hyperparameters.items()})

    train_image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        model_id=model_id,
        model_version=model_version,
        image_scope="training",
        instance_type=instance_type,
    )
    train_model_uri = model_uris.retrieve(
        model_id=model_id, model_version=model_version, model_scope="training"
    )
    patched_source_dir = download_and_patch_script(model_id, model_version, source_dir)

    estimator = Estimator(
        role=role,
        image_uri=train_image_uri,
        source_dir=patched_source_dir,      # AWS's script, patched for reduce_retracing
        entry_point="transfer_learning.py",
        model_uri=train_model_uri,          # pretrained EfficientDet D1 weights
        instance_count=1,
        instance_type=instance_type,
        max_run=360000,
        hyperparameters=hyperparameters,
        output_path=output_path,
    )
    estimator.fit({"training": training_s3_uri})
    return estimator


def deploy(estimator, instance_type="ml.m5.xlarge"):
    """Deploy the fine-tuned model to a real-time endpoint. Remember to delete it after."""
    return estimator.deploy(instance_type=instance_type)
