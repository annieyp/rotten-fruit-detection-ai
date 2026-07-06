import shutil
import subprocess
import sys
from pathlib import Path

from dataset import load_data
from model import build_model


TF_MODELS_DIR = Path("tensorflow_models")


def run(command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def object_detection_dir():
    """Clone tensorflow/models and install the Object Detection API if needed."""
    research_dir = TF_MODELS_DIR / "research"

    if not TF_MODELS_DIR.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/tensorflow/models.git", str(TF_MODELS_DIR)])
        protos = [str(p.relative_to(research_dir))
                  for p in (research_dir / "object_detection" / "protos").glob("*.proto")]
        run(["protoc", *protos, "--python_out=."], cwd=research_dir)
        shutil.copy2(
            research_dir / "object_detection" / "packages" / "tf2" / "setup.py",
            research_dir / "setup.py",
        )
        run([sys.executable, "-m", "pip", "install", "-q", "."], cwd=research_dir)

    return research_dir


def train(pipeline_config_path, model_dir, num_steps=5000):
    research_dir = object_detection_dir()
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, str(research_dir / "object_detection" / "model_main_tf2.py"),
        f"--pipeline_config_path={pipeline_config_path}",
        f"--model_dir={model_dir}",
        f"--num_train_steps={num_steps}",
        "--alsologtostderr",
    ])


def test_model(pipeline_config_path, model_dir):
    research_dir = object_detection_dir()

    run([
        sys.executable, str(research_dir / "object_detection" / "model_main_tf2.py"),
        f"--pipeline_config_path={pipeline_config_path}",
        f"--model_dir={model_dir}",
        f"--checkpoint_dir={model_dir}",
        "--eval_timeout=1",
        "--alsologtostderr",
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
    output_dir="artifacts",
    model_dir="training/ssd_efficientdet_d5",
    export_dir="artifacts/ssd_efficientdet_d5_saved_model",
    num_steps=5000,
    batch_size=2,
):
    record_paths, class_names = load_data(data_dir, output_dir)
    print("Classes:", class_names)

    pipeline_config_path = build_model(
        num_classes=len(class_names),
        record_paths=record_paths,
        output_dir=output_dir,
        batch_size=batch_size,
        num_steps=num_steps,
    )

    train(pipeline_config_path, model_dir, num_steps=num_steps)
    test_model(pipeline_config_path, model_dir)
    export_model(pipeline_config_path, model_dir, export_dir)

    print("Saved model:", export_dir)


if __name__ == "__main__":
    main()
