import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from dataset import prepare_detection_dataset


DEFAULT_PRETRAINED_MODEL_NAME = "efficientdet_d5_coco17_tpu-32"
DEFAULT_PRETRAINED_MODEL_URL = (
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/"
    "efficientdet_d5_coco17_tpu-32.tar.gz"
)


def prepare_training_data(
    data_dir="dataset",
    output_dir="artifacts",
    export_tensorflow_records=True,
):
    return prepare_detection_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        export_tensorflow_records=export_tensorflow_records,
    )


def run_command(command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def ensure_tensorflow_models_repo(tf_models_dir):
    tf_models_dir = Path(tf_models_dir)

    if not tf_models_dir.exists():
        run_command(
            [
                "git",
                "clone",
                "https://github.com/tensorflow/models.git",
                str(tf_models_dir),
            ]
        )

    return tf_models_dir / "research"


def install_tensorflow_object_detection_api(tfod_research_dir):
    tfod_research_dir = Path(tfod_research_dir)
    setup_source = tfod_research_dir / "object_detection" / "packages" / "tf2" / "setup.py"
    setup_target = tfod_research_dir / "setup.py"
    proto_files = sorted(
        str(path.relative_to(tfod_research_dir))
        for path in (tfod_research_dir / "object_detection" / "protos").glob("*.proto")
    )

    run_command(["protoc", *proto_files, "--python_out=."], cwd=tfod_research_dir)
    shutil.copy2(setup_source, setup_target)
    run_command([sys.executable, "-m", "pip", "install", "-q", "."], cwd=tfod_research_dir)


def download_pretrained_model(model_name, model_url, pretrained_models_dir):
    pretrained_models_dir = Path(pretrained_models_dir)
    pretrained_models_dir.mkdir(parents=True, exist_ok=True)
    pretrained_model_dir = pretrained_models_dir / model_name
    archive_path = pretrained_models_dir / f"{model_name}.tar.gz"

    if not pretrained_model_dir.exists():
        urllib.request.urlretrieve(model_url, archive_path)
        with tarfile.open(archive_path) as archive:
            archive.extractall(pretrained_models_dir)

    return pretrained_model_dir


def patch_pipeline_config(
    template_config_path,
    output_config_path,
    num_classes,
    batch_size,
    num_train_steps,
    fine_tune_checkpoint,
    label_map_path,
    train_record_path,
    eval_record_path,
):
    config = Path(template_config_path).read_text()
    config = re.sub(r"num_classes: \d+", f"num_classes: {num_classes}", config, count=1)
    config = re.sub(r"batch_size: \d+", f"batch_size: {batch_size}", config, count=1)
    config = re.sub(r"num_steps: \d+", f"num_steps: {num_train_steps}", config, count=1)
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
    config = re.sub(
        r'label_map_path: ".*?"',
        f'label_map_path: "{label_map_path}"',
        config,
    )

    input_paths = [str(train_record_path), str(eval_record_path)]

    def replace_input_path(match):
        if input_paths:
            return f'input_path: "{input_paths.pop(0)}"'
        return match.group(0)

    config = re.sub(r'input_path: ".*?"', replace_input_path, config)

    output_config_path = Path(output_config_path)
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(config)
    return output_config_path


def train(
    data_dir="dataset",
    output_dir="artifacts",
    tf_models_dir="tensorflow_models",
    pretrained_models_dir="pretrained_models",
    pretrained_model_name=DEFAULT_PRETRAINED_MODEL_NAME,
    pretrained_model_url=DEFAULT_PRETRAINED_MODEL_URL,
    model_dir="training/ssd_efficientdet_d5",
    pipeline_config_path=None,
    num_train_steps=5000,
    batch_size=2,
    install_api=True,
):
    dataset, summary = prepare_training_data(
        data_dir=data_dir,
        output_dir=output_dir,
        export_tensorflow_records=True,
    )

    output_dir = Path(output_dir)
    tfod_research_dir = ensure_tensorflow_models_repo(tf_models_dir)

    if install_api:
        install_tensorflow_object_detection_api(tfod_research_dir)

    pretrained_model_dir = download_pretrained_model(
        model_name=pretrained_model_name,
        model_url=pretrained_model_url,
        pretrained_models_dir=pretrained_models_dir,
    )

    eval_record_path = output_dir / "valid.record"
    if not eval_record_path.exists():
        eval_record_path = output_dir / "test.record"

    pipeline_config_path = patch_pipeline_config(
        template_config_path=pretrained_model_dir / "pipeline.config",
        output_config_path=pipeline_config_path
        or output_dir / "ssd_efficientdet_d5_pipeline.config",
        num_classes=len(dataset["class_names"]),
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        fine_tune_checkpoint=pretrained_model_dir / "checkpoint" / "ckpt-0",
        label_map_path=output_dir / "label_map.pbtxt",
        train_record_path=output_dir / "train.record",
        eval_record_path=eval_record_path,
    )

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            sys.executable,
            str(tfod_research_dir / "object_detection" / "model_main_tf2.py"),
            f"--pipeline_config_path={pipeline_config_path}",
            f"--model_dir={model_dir}",
            "--alsologtostderr",
            f"--num_train_steps={num_train_steps}",
        ]
    )

    return {
        "dataset": dataset,
        "summary": summary,
        "pipeline_config_path": pipeline_config_path,
        "model_dir": model_dir,
        "tfod_research_dir": tfod_research_dir,
    }


def test_model(
    pipeline_config_path="artifacts/ssd_efficientdet_d5_pipeline.config",
    model_dir="training/ssd_efficientdet_d5",
    tf_models_dir="tensorflow_models",
    num_eval_steps=500,
    install_api=False,
):
    tfod_research_dir = ensure_tensorflow_models_repo(tf_models_dir)

    if install_api:
        install_tensorflow_object_detection_api(tfod_research_dir)

    run_command(
        [
            sys.executable,
            str(tfod_research_dir / "object_detection" / "model_main_tf2.py"),
            f"--pipeline_config_path={pipeline_config_path}",
            f"--model_dir={model_dir}",
            f"--checkpoint_dir={model_dir}",
            "--alsologtostderr",
            "--eval_timeout=1",
            f"--num_eval_steps={num_eval_steps}",
        ]
    )

    return {
        "pipeline_config_path": Path(pipeline_config_path),
        "model_dir": Path(model_dir),
        "tfod_research_dir": tfod_research_dir,
    }


def export_model(
    pipeline_config_path="artifacts/ssd_efficientdet_d5_pipeline.config",
    model_dir="training/ssd_efficientdet_d5",
    export_dir="artifacts/ssd_efficientdet_d5_saved_model",
    tf_models_dir="tensorflow_models",
):
    tfod_research_dir = ensure_tensorflow_models_repo(tf_models_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            sys.executable,
            str(tfod_research_dir / "object_detection" / "exporter_main_v2.py"),
            "--input_type=image_tensor",
            f"--pipeline_config_path={pipeline_config_path}",
            f"--trained_checkpoint_dir={model_dir}",
            f"--output_directory={export_dir}",
        ]
    )

    return export_dir


def train_evaluate_export(
    data_dir="dataset",
    output_dir="artifacts",
    tf_models_dir="tensorflow_models",
    pretrained_models_dir="pretrained_models",
    model_dir="training/ssd_efficientdet_d5",
    export_dir="artifacts/ssd_efficientdet_d5_saved_model",
    num_train_steps=5000,
    num_eval_steps=500,
    batch_size=2,
    install_api=True,
):
    result = train(
        data_dir=data_dir,
        output_dir=output_dir,
        tf_models_dir=tf_models_dir,
        pretrained_models_dir=pretrained_models_dir,
        model_dir=model_dir,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        install_api=install_api,
    )
    evaluation = test_model(
        pipeline_config_path=result["pipeline_config_path"],
        model_dir=result["model_dir"],
        tf_models_dir=tf_models_dir,
        num_eval_steps=num_eval_steps,
        install_api=False,
    )
    exported_model_dir = export_model(
        pipeline_config_path=result["pipeline_config_path"],
        model_dir=result["model_dir"],
        export_dir=export_dir,
        tf_models_dir=tf_models_dir,
    )

    return {
        **result,
        "evaluation": evaluation,
        "export_dir": exported_model_dir,
    }


def main():
    train_evaluate_export()


if __name__ == "__main__":
    main()
