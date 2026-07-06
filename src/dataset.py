import csv
import json
from collections import Counter, OrderedDict
from pathlib import Path

import tensorflow as tf


SPLITS = ("train", "valid", "test")


def get_split_dir(data_dir, split_name):
    for item in Path(data_dir).iterdir():
        if item.is_dir() and item.name.lower() == split_name:
            return item

    return None


def get_annotations_csv(split_dir):
    for name in ("annotations.csv", "_annotations.csv"):
        csv_path = split_dir / name
        if csv_path.exists():
            return csv_path

    csv_files = sorted(split_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise ValueError(f"Could not find an annotations CSV inside {split_dir}.")


def normalize_class_name(class_name):
    return class_name.strip().lower()


def condition_from_class_name(class_name):
    class_name = normalize_class_name(class_name)

    if class_name.startswith(("good", "fresh")):
        return "fresh"
    if class_name.startswith(("bad", "rotten")):
        return "rotten"

    return "unknown"


def fruit_from_class_name(class_name):
    class_name = normalize_class_name(class_name)
    words = class_name.split()

    if len(words) >= 2:
        return " ".join(words[1:])

    return class_name


def image_path_for(split_dir, filename):
    candidates = [
        split_dir / filename,
        split_dir / "images" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(split_dir.rglob(filename))
    if matches:
        return matches[0]

    raise ValueError(f"Could not find image '{filename}' inside {split_dir}.")


def read_split_annotations(split_dir):
    annotations_path = get_annotations_csv(split_dir)
    records = []

    with annotations_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "filename",
            "width",
            "height",
            "class",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"{annotations_path} is missing columns: {sorted(missing_columns)}"
            )

        for row in reader:
            class_name = normalize_class_name(row["class"])
            image_path = image_path_for(split_dir, row["filename"])

            records.append(
                {
                    "filename": row["filename"],
                    "image_path": str(image_path),
                    "width": int(float(row["width"])),
                    "height": int(float(row["height"])),
                    "class_name": class_name,
                    "condition": condition_from_class_name(class_name),
                    "fruit": fruit_from_class_name(class_name),
                    "bbox": {
                        "xmin": float(row["xmin"]),
                        "ymin": float(row["ymin"]),
                        "xmax": float(row["xmax"]),
                        "ymax": float(row["ymax"]),
                    },
                }
            )

    return records


def collect_detection_dataset(data_dir):
    data_dir = Path(data_dir).resolve()
    split_records = {}
    class_names = OrderedDict()

    for split_name in SPLITS:
        split_dir = get_split_dir(data_dir, split_name)
        if split_dir is None:
            continue

        records = read_split_annotations(split_dir)
        split_records[split_name] = records

        for record in records:
            class_names.setdefault(record["class_name"], None)

    if "train" not in split_records:
        raise ValueError(f"Could not find a train split inside {data_dir}.")
    if "test" not in split_records:
        raise ValueError(f"Could not find a test split inside {data_dir}.")
    if not class_names:
        raise ValueError(f"No annotations found inside {data_dir}.")

    return {
        "data_dir": str(data_dir),
        "class_names": list(class_names.keys()),
        "splits": split_records,
    }


def summarize_detection_dataset(dataset):
    summary = {
        "data_dir": dataset["data_dir"],
        "class_names": dataset["class_names"],
        "num_classes": len(dataset["class_names"]),
        "splits": {},
    }

    for split_name, records in dataset["splits"].items():
        images = {record["filename"] for record in records}
        class_counts = Counter(record["class_name"] for record in records)
        condition_counts = Counter(record["condition"] for record in records)
        fruit_counts = Counter(record["fruit"] for record in records)

        summary["splits"][split_name] = {
            "images": len(images),
            "annotations": len(records),
            "classes": dict(sorted(class_counts.items())),
            "conditions": dict(sorted(condition_counts.items())),
            "fruits": dict(sorted(fruit_counts.items())),
        }

    return summary


def write_detection_metadata(dataset, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_detection_dataset(dataset)
    (output_dir / "class_names.txt").write_text("\n".join(dataset["class_names"]) + "\n")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    for split_name, records in dataset["splits"].items():
        split_path = output_dir / f"{split_name}_annotations.json"
        split_path.write_text(json.dumps(records, indent=2))

    return summary


def write_label_map_pbtxt(class_names, output_path):
    lines = []

    for index, class_name in enumerate(class_names, start=1):
        escaped_name = class_name.replace("'", "\\'")
        lines.extend(
            [
                "item {",
                f"  id: {index}",
                f"  name: '{escaped_name}'",
                "}",
                "",
            ]
        )

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines))
    return output_path


def bytes_feature(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_list_feature(values):
    encoded_values = [
        value.encode("utf-8") if isinstance(value, str) else value for value in values
    ]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_values))


def grouped_records(records):
    grouped = OrderedDict()

    for record in records:
        grouped.setdefault(record["image_path"], []).append(record)

    return grouped


def tf_example_for_image(image_records, class_to_id):
    first_record = image_records[0]
    width = first_record["width"]
    height = first_record["height"]
    filename = first_record["filename"]
    image_path = Path(first_record["image_path"])
    encoded_image = image_path.read_bytes()
    image_format = image_path.suffix.lower().lstrip(".").encode("utf-8")

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    class_texts = []
    class_ids = []

    for record in image_records:
        bbox = record["bbox"]
        xmins.append(max(0.0, min(1.0, bbox["xmin"] / width)))
        xmaxs.append(max(0.0, min(1.0, bbox["xmax"] / width)))
        ymins.append(max(0.0, min(1.0, bbox["ymin"] / height)))
        ymaxs.append(max(0.0, min(1.0, bbox["ymax"] / height)))
        class_texts.append(record["class_name"])
        class_ids.append(class_to_id[record["class_name"]])

    features = {
        "image/height": int64_feature(height),
        "image/width": int64_feature(width),
        "image/filename": bytes_feature(filename),
        "image/source_id": bytes_feature(filename),
        "image/encoded": bytes_feature(encoded_image),
        "image/format": bytes_feature(image_format),
        "image/object/bbox/xmin": float_list_feature(xmins),
        "image/object/bbox/xmax": float_list_feature(xmaxs),
        "image/object/bbox/ymin": float_list_feature(ymins),
        "image/object/bbox/ymax": float_list_feature(ymaxs),
        "image/object/class/text": bytes_list_feature(class_texts),
        "image/object/class/label": int64_list_feature(class_ids),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(records, class_to_id, output_path):
    output_path = Path(output_path)
    grouped = grouped_records(records)

    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for image_records in grouped.values():
            writer.write(tf_example_for_image(image_records, class_to_id).SerializeToString())

    return output_path


def write_tensorflow_object_detection_files(dataset, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {
        class_name: index for index, class_name in enumerate(dataset["class_names"], start=1)
    }
    label_map_path = write_label_map_pbtxt(
        dataset["class_names"],
        output_dir / "label_map.pbtxt",
    )
    tfrecord_paths = {}

    for split_name, records in dataset["splits"].items():
        tfrecord_paths[split_name] = write_tfrecord(
            records,
            class_to_id,
            output_dir / f"{split_name}.record",
        )

    return label_map_path, tfrecord_paths


def prepare_detection_dataset(
    data_dir,
    output_dir=None,
    export_tensorflow_records=False,
):
    dataset = collect_detection_dataset(data_dir)

    if output_dir is not None:
        summary = write_detection_metadata(dataset, output_dir)
        if export_tensorflow_records:
            label_map_path, tfrecord_paths = write_tensorflow_object_detection_files(
                dataset,
                output_dir,
            )
            summary["tensorflow_object_detection"] = {
                "label_map": str(label_map_path),
                "tfrecords": {
                    split_name: str(path)
                    for split_name, path in tfrecord_paths.items()
                },
            }
    else:
        summary = summarize_detection_dataset(dataset)

    return dataset, summary
