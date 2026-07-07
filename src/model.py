import re
import tarfile
import urllib.request
from pathlib import Path


ZOO_BASE_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/"

# EfficientDet variants from the TF2 Detection Model Zoo. D1 is the cost/accuracy
# sweet spot; D0 is cheapest, D5 is most accurate but slow/expensive.
MODELS = {
    "efficientdet_d0": "efficientdet_d0_coco17_tpu-32",
    "efficientdet_d1": "efficientdet_d1_coco17_tpu-32",
    "efficientdet_d5": "efficientdet_d5_coco17_tpu-32",
}
DEFAULT_MODEL = "efficientdet_d1"


def read_label_map_names(label_map_path):
    text = Path(label_map_path).read_text()
    return re.findall(r"name:\s*'([^']*)'", text)


def download_pretrained_model(model_name=DEFAULT_MODEL, pretrained_models_dir="pre-trained-models"):
    zoo_name = MODELS[model_name]
    pretrained_models_dir = Path(pretrained_models_dir)
    pretrained_models_dir.mkdir(parents=True, exist_ok=True)
    model_dir = pretrained_models_dir / zoo_name

    if not model_dir.exists():
        archive_path = pretrained_models_dir / f"{zoo_name}.tar.gz"
        urllib.request.urlretrieve(ZOO_BASE_URL + f"{zoo_name}.tar.gz", archive_path)
        with tarfile.open(archive_path) as archive:
            archive.extractall(pretrained_models_dir)

    return model_dir


def build_model(
    label_map_path="annotations/label_map.pbtxt",
    train_record_path="annotations/train.record",
    eval_record_path="annotations/test.record",
    model_dir="models/efficientdet_d1",
    model_name=DEFAULT_MODEL,
    pretrained_models_dir="pre-trained-models",
    batch_size=8,
    num_steps=4000,
):
    """Download the chosen EfficientDet variant and write model_dir/pipeline.config."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pretrained_dir = download_pretrained_model(model_name, pretrained_models_dir)
    fine_tune_checkpoint = pretrained_dir / "checkpoint" / "ckpt-0"
    num_classes = len(read_label_map_names(label_map_path))

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
    config = re.sub(r"use_bfloat16: (?:true|false)", "use_bfloat16: false", config)
    config = re.sub(r'label_map_path: ".*?"', f'label_map_path: "{label_map_path}"', config)

    # First input_path is train_input_reader, second is eval_input_reader.
    input_paths = [str(train_record_path), str(eval_record_path)]
    config = re.sub(
        r'input_path: ".*?"',
        lambda match: f'input_path: "{input_paths.pop(0)}"' if input_paths else match.group(0),
        config,
    )

    pipeline_config_path = model_dir / "pipeline.config"
    pipeline_config_path.write_text(config)
    print(
        f"Wrote pipeline config: {pipeline_config_path} "
        f"(model={model_name}, num_classes={num_classes}, batch_size={batch_size}, steps={num_steps})"
    )
    return pipeline_config_path
