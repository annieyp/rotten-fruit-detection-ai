import shutil
import subprocess
import sys
from pathlib import Path

from dataset import load_data
from model import DEFAULT_MODEL, build_model


# Cloned tensorflow/models repo (the Object Detection API lives in research/).
API_REPO_DIR = Path("tensorflow_models")


def run(command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def object_detection_dir():
    """Clone tensorflow/models and install the Object Detection API if needed."""
    research_dir = API_REPO_DIR / "research"

    if not API_REPO_DIR.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/tensorflow/models.git", str(API_REPO_DIR)])
        protos = [str(p.relative_to(research_dir))
                  for p in (research_dir / "object_detection" / "protos").glob("*.proto")]
        run(["protoc", *protos, "--python_out=."], cwd=research_dir)
        shutil.copy2(
            research_dir / "object_detection" / "packages" / "tf2" / "setup.py",
            research_dir / "setup.py",
        )
        run([sys.executable, "-m", "pip", "install", "-q", "."], cwd=research_dir)

    return research_dir


def train(pipeline_config_path, model_dir):
    # model_main_tf2.py auto-uses MirroredStrategy across ALL local GPUs, so a
    # multi-GPU instance is utilised without any extra flag.
    research_dir = object_detection_dir()
    run([
        sys.executable, str(research_dir / "object_detection" / "model_main_tf2.py"),
        f"--model_dir={model_dir}",
        f"--pipeline_config_path={pipeline_config_path}",
    ])


def test_model(pipeline_config_path, model_dir):
    # --eval_timeout=1 makes eval score the final checkpoint once and exit,
    # instead of blocking ~1h waiting for new checkpoints (default 3600s).
    research_dir = object_detection_dir()
    run([
        sys.executable, str(research_dir / "object_detection" / "model_main_tf2.py"),
        f"--model_dir={model_dir}",
        f"--pipeline_config_path={pipeline_config_path}",
        f"--checkpoint_dir={model_dir}",
        "--eval_timeout=1",
    ])


def export_model(pipeline_config_path, model_dir, export_dir):
    research_dir = object_detection_dir()
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, str(research_dir / "object_detection" / "exporter_main_v2.py"),
        "--input_type=image_tensor",
        f"--pipeline_config_path={pipeline_config_path}",
        f"--trained_checkpoint_dir={model_dir}",
        f"--output_directory={export_dir}",
    ])


def main(
    data_dir="dataset",
    annotations_dir="annotations",
    model_name=DEFAULT_MODEL,
    num_steps=4000,
    batch_size=8,
):
    model_dir = f"models/{model_name}"
    export_dir = f"exported-models/{model_name}"

    # Step 1: build label_map.pbtxt + TFRecords from the raw dataset/ CSV exports.
    record_paths, label_map_path, class_names = load_data(data_dir, annotations_dir)
    eval_record_path = record_paths.get("valid", record_paths.get("test"))

    # Step 2: download the pretrained model and configure the pipeline for our classes.
    pipeline_config_path = build_model(
        label_map_path=label_map_path,
        train_record_path=record_paths["train"],
        eval_record_path=eval_record_path,
        model_dir=model_dir,
        model_name=model_name,
        batch_size=batch_size,
        num_steps=num_steps,
    )

    train(pipeline_config_path, model_dir)
    test_model(pipeline_config_path, model_dir)
    export_model(pipeline_config_path, model_dir, export_dir)

    print("Exported model:", export_dir)


if __name__ == "__main__":
    main()
