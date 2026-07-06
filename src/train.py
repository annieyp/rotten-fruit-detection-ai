from dataset import prepare_detection_dataset


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


def train(*args, **kwargs):
    raise NotImplementedError(
        "Training is intentionally model-agnostic right now. "
        "For SSD EfficientDet D5, use the generated TFRecords and label_map.pbtxt "
        "with the TensorFlow Object Detection API pipeline config."
    )


def test_model(*args, **kwargs):
    raise NotImplementedError(
        "Evaluation is intentionally model-agnostic right now. "
        "Use the evaluation command for your selected detection framework."
    )


def main():
    _, summary = prepare_training_data()
    print("Prepared object detection dataset.")
    print(summary)


if __name__ == "__main__":
    main()
