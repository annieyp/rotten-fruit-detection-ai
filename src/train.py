from dataset import prepare_detection_dataset


def prepare_training_data(data_dir="dataset", output_dir="artifacts"):
    return prepare_detection_dataset(data_dir=data_dir, output_dir=output_dir)


def train(*args, **kwargs):
    raise NotImplementedError(
        "Training is intentionally model-agnostic right now. "
        "Pick an object detection model/framework first, then implement train()."
    )


def test_model(*args, **kwargs):
    raise NotImplementedError(
        "Evaluation is intentionally model-agnostic right now. "
        "Pick an object detection model/framework first, then implement test_model()."
    )


def main():
    _, summary = prepare_training_data()
    print("Prepared object detection dataset.")
    print(summary)


if __name__ == "__main__":
    main()
