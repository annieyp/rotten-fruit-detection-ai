import csv
from collections import OrderedDict
from pathlib import Path

import keras
import tensorflow as tf


SPLITS = ("train", "valid", "test")

# Columns in a Roboflow "TensorFlow Object Detection" CSV export (absolute pixels).
CSV_COLUMNS = ("filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax")

# KerasHub RetinaNet works in [y_min, x_min, y_max, x_max] order.
BBOX_FORMAT = "yxyx"


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


def build_split_dataset(split_dir, rows, class_to_id):
    boxes_by_image = OrderedDict()
    for row in rows:
        boxes_by_image.setdefault(row["filename"], []).append(row)

    paths, boxes, labels = [], [], []
    for filename, items in boxes_by_image.items():
        paths.append(str(split_dir / filename))
        boxes.append([[b["ymin"], b["xmin"], b["ymax"], b["xmax"]] for b in items])
        labels.append([class_to_id[b["class"]] for b in items])

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(paths),
            tf.ragged.constant(boxes, dtype=tf.float32, ragged_rank=1),
            tf.ragged.constant(labels, dtype=tf.int32),
        )
    )

    def load(path, image_boxes, image_labels):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        return {
            "images": image,
            "bounding_boxes": {"boxes": image_boxes, "labels": image_labels},
        }

    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)


def to_tuple(record):
    return record["images"], {
        "boxes": record["bounding_boxes"]["boxes"],
        "labels": record["bounding_boxes"]["labels"],
    }


def load_data(data_dir="dataset", image_size=640, batch_size=4, max_boxes=100):
    """Read the Roboflow CSV export into batched RetinaNet-ready tf.data pipelines."""
    data_dir = Path(data_dir)

    rows_by_split = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if split_dir.is_dir():
            rows_by_split[split] = read_annotations(split_dir)

    if "train" not in rows_by_split:
        raise ValueError(f"Could not find a train split inside {data_dir}.")

    class_names = collect_class_names(rows_by_split)
    class_to_id = {name: index for index, name in enumerate(class_names)}  # 0-indexed

    resizing = keras.layers.Resizing(
        height=image_size,
        width=image_size,
        pad_to_aspect_ratio=True,
        bounding_box_format=BBOX_FORMAT,
    )
    max_boxes_layer = keras.layers.MaxNumBoundingBoxes(
        max_number=max_boxes,
        bounding_box_format=BBOX_FORMAT,
    )

    datasets = {}
    for split, rows in rows_by_split.items():
        dataset = build_split_dataset(data_dir / split, rows, class_to_id)
        dataset = dataset.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(max_boxes_layer, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        datasets[split] = dataset.prefetch(tf.data.AUTOTUNE)
        print(f"{split}: {len(rows)} boxes across {len(dataset) * batch_size} images (approx)")

    print("Classes:", class_names)
    return datasets, class_names
