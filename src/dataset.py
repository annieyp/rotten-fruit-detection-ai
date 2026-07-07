import csv
from collections import OrderedDict
from pathlib import Path

import tensorflow as tf


SPLITS = ("train", "valid", "test")

# Columns in a Roboflow "TensorFlow Object Detection" CSV export.
CSV_COLUMNS = ("filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax")


def find_annotations_csv(split_dir):
    for name in ("_annotations.csv", "annotations.csv"):
        csv_path = split_dir / name
        if csv_path.exists():
            return csv_path

    matches = sorted(split_dir.glob("*.csv"))
    if matches:
        return matches[0]

    raise ValueError(f"Could not find an annotations CSV inside {split_dir}.")


def read_annotations(split_dir):
    csv_path = find_annotations_csv(split_dir)

    rows = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                {
                    "filename": row["filename"],
                    "width": int(float(row["width"])),
                    "height": int(float(row["height"])),
                    "class": row["class"].strip(),
                    "xmin": float(row["xmin"]),
                    "ymin": float(row["ymin"]),
                    "xmax": float(row["xmax"]),
                    "ymax": float(row["ymax"]),
                }
            )

    return rows


def collect_class_names(rows_by_split):
    class_names = OrderedDict()
    for rows in rows_by_split.values():
        for row in sorted(rows, key=lambda r: r["class"]):
            class_names.setdefault(row["class"], None)
    return list(class_names.keys())


def write_label_map(class_names, output_path):
    # label_map.pbtxt, ids starting at 1 (Object Detection API convention).
    lines = []
    for class_id, class_name in enumerate(class_names, start=1):
        lines += ["item {", f"    id: {class_id}", f"    name: '{class_name}'", "}", ""]
    output_path.write_text("\n".join(lines))


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


def create_tf_example(image_dir, filename, boxes, class_to_id):
    # One tf.Example per image, holding every bounding box for that image.
    image_path = image_dir / filename
    encoded_image = image_path.read_bytes()
    image_format = image_path.suffix.lower().lstrip(".").encode("utf-8")

    width = boxes[0]["width"]
    height = boxes[0]["height"]

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
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
                "image/object/class/text": bytes_list_feature([b["class"] for b in boxes]),
                "image/object/class/label": int64_list_feature(
                    [class_to_id[b["class"]] for b in boxes]
                ),
            }
        )
    )
    return example


def generate_tfrecord(image_dir, rows, class_to_id, output_path):
    boxes_by_image = OrderedDict()
    for row in rows:
        boxes_by_image.setdefault(row["filename"], []).append(row)

    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for filename, boxes in boxes_by_image.items():
            writer.write(
                create_tf_example(image_dir, filename, boxes, class_to_id).SerializeToString()
            )

    return len(boxes_by_image)


def load_data(data_dir="dataset", annotations_dir="annotations"):
    """Build label_map.pbtxt and one .record file per split, tutorial-style."""
    data_dir = Path(data_dir)
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if split_dir.is_dir():
            rows_by_split[split] = read_annotations(split_dir)

    if "train" not in rows_by_split:
        raise ValueError(f"Could not find a train split inside {data_dir}.")

    class_names = collect_class_names(rows_by_split)
    class_to_id = {name: index for index, name in enumerate(class_names, start=1)}

    label_map_path = annotations_dir / "label_map.pbtxt"
    write_label_map(class_names, label_map_path)

    record_paths = {}
    for split, rows in rows_by_split.items():
        record_path = annotations_dir / f"{split}.record"
        num_images = generate_tfrecord(data_dir / split, rows, class_to_id, record_path)
        record_paths[split] = record_path
        print(f"{split}: {len(rows)} boxes across {num_images} images")

    print("Classes:", class_names)
    return record_paths, label_map_path, class_names
