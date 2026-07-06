import re
import tarfile
import urllib.request
from pathlib import Path

import tensorflow as tf


PRETRAINED_MODEL_NAME = "efficientdet_d5_coco17_tpu-32"
PRETRAINED_MODEL_URL = (
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/"
    "efficientdet_d5_coco17_tpu-32.tar.gz"
)


def download_pretrained_model(pretrained_models_dir="pretrained_models"):
    pretrained_models_dir = Path(pretrained_models_dir)
    pretrained_models_dir.mkdir(parents=True, exist_ok=True)
    model_dir = pretrained_models_dir / PRETRAINED_MODEL_NAME

    if not model_dir.exists():
        archive_path = pretrained_models_dir / f"{PRETRAINED_MODEL_NAME}.tar.gz"
        urllib.request.urlretrieve(PRETRAINED_MODEL_URL, archive_path)
        with tarfile.open(archive_path) as archive:
            archive.extractall(pretrained_models_dir)

    return model_dir


def build_model(
    num_classes,
    record_paths,
    output_dir="artifacts",
    batch_size=2,
    num_steps=5000,
    pretrained_models_dir="pretrained_models",
):
    """Download EfficientDet D5 and write a pipeline.config fine-tuned for our classes."""
    output_dir = Path(output_dir)
    pretrained_dir = download_pretrained_model(pretrained_models_dir)

    label_map_path = output_dir / "label_map.pbtxt"
    fine_tune_checkpoint = pretrained_dir / "checkpoint" / "ckpt-0"
    eval_record = record_paths.get("valid", record_paths.get("test"))

    config = (pretrained_dir / "pipeline.config").read_text()
    config = re.sub(r"num_classes: \d+", f"num_classes: {num_classes}", config, count=1)
    config = re.sub(r"batch_size: \d+", f"batch_size: {batch_size}", config, count=1)
    config = re.sub(r"num_steps: \d+", f"num_steps: {num_steps}", config, count=1)
    config = re.sub(
        r'fine_tune_checkpoint: ".*?"',
        f'fine_tune_checkpoint: "{fine_tune_checkpoint}"',
        config,
    )
    config = re.sub(
        r'fine_tune_checkpoint_type: ".*?"',
        'fine_tune_checkpoint_type: "detection"',
        config,
    )
    config = re.sub(r'label_map_path: ".*?"', f'label_map_path: "{label_map_path}"', config)

    input_paths = [str(record_paths["train"]), str(eval_record)]
    config = re.sub(
        r'input_path: ".*?"',
        lambda match: f'input_path: "{input_paths.pop(0)}"' if input_paths else match.group(0),
        config,
    )

    pipeline_config_path = output_dir / "ssd_efficientdet_d5_pipeline.config"
    pipeline_config_path.write_text(config)
    return pipeline_config_path


def load_model(export_dir):
    saved_model = tf.saved_model.load(str(Path(export_dir) / "saved_model"))
    return saved_model.signatures["serving_default"]
