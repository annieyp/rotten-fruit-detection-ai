import csv
from pathlib import Path

import tensorflow as tf


SPLITS = ("train", "valid", "test")

# Object detection collapses every annotation into two classes: fresh vs rotten.
# TF Object Detection API expects class ids to start at 1.
CLASS_NAMES = ("fresh", "rotten")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES, start=1)}


def condition_from_class_name(class_name):
    name = class_name.strip().lower()

    if name.startswith(("good", "fresh")):
        return "fresh"
    if name.startswith(("bad", "rotten")):
        return "rotten"

    return None


def read_annotations(split_dir):
    csv_path = split_dir / "annotations.csv"
    if not csv_path.exists():
        raise ValueError(f"Could not find annotations.csv inside {split_dir}.")

    rows = []
    with csv_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            condition = condition_from_class_name(row["class"])
            if condition is None:
                continue

            rows.append(
                {
                    "filename": row["filename"],
                    "width": int(float(row["width"])),
                    "height": int(float(row["height"])),
                    "condition": condition,
                    "xmin": float(row["xmin"]),
                    "ymin": float(row["ymin"]),
                    "xmax": float(row["xmax"]),
                    "ymax": float(row["ymax"]),
                }
            )

    return rows


def bytes_feature(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_list_feature(values):
    encoded = [value.encode("utf-8") for value in values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded))


def build_example(split_dir, filename, boxes):
    image_path = split_dir / filename
    encoded_image = image_path.read_bytes()
    image_format = image_path.suffix.lower().lstrip(".")

    width = boxes[0]["width"]
    height = boxes[0]["height"]

    feature = {
        "image/height": int64_feature(height),
        "image/width": int64_feature(width),
        "image/filename": bytes_feature(filename),
        "image/source_id": bytes_feature(filename),
        "image/encoded": bytes_feature(encoded_image),
        "image/format": bytes_feature(image_format),
        "image/object/bbox/xmin": float_list_feature([b["xmin"] / width for b in boxes]),
        "image/object/bbox/xmax": float_list_feature([b["xmax"] / width for b in boxes]),
        "image/object/bbox/ymin": float_list_feature([b["ymin"] / height for b in boxes]),
        "image/object/bbox/ymax": float_list_feature([b["ymax"] / height for b in boxes]),
        "image/object/class/text": bytes_list_feature([b["condition"] for b in boxes]),
        "image/object/class/label": int64_list_feature(
            [CLASS_TO_ID[b["condition"]] for b in boxes]
        ),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(split_dir, rows, output_path):
    boxes_by_image = {}
    for row in rows:
        boxes_by_image.setdefault(row["filename"], []).append(row)

    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for filename, boxes in boxes_by_image.items():
            writer.write(build_example(split_dir, filename, boxes).SerializeToString())

    return len(boxes_by_image)


def write_label_map(output_path):
    lines = []
    for class_name, class_id in CLASS_TO_ID.items():
        lines += ["item {", f"  id: {class_id}", f"  name: '{class_name}'", "}", ""]
    output_path.write_text("\n".join(lines))


def load_data(data_dir="dataset", output_dir="artifacts"):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_label_map(output_dir / "label_map.pbtxt")

    record_paths = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue

        rows = read_annotations(split_dir)
        record_path = output_dir / f"{split}.record"
        num_images = write_tfrecord(split_dir, rows, record_path)
        record_paths[split] = record_path
        print(f"{split}: {len(rows)} boxes across {num_images} images")

    if "train" not in record_paths:
        raise ValueError(f"Could not find a train split inside {data_dir}.")

    return record_paths, list(CLASS_NAMES)
