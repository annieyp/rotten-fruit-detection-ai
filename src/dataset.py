import csv
import shutil
from collections import OrderedDict
from pathlib import Path

import yaml


SPLITS = ("train", "valid", "test")

# Columns in a Roboflow "TensorFlow Object Detection" CSV export (absolute pixels).
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


def prepare_yolo_dataset(data_dir="dataset", output_dir="yolo_data", splits=("train", "valid", "test")):
    """Convert a Roboflow TF CSV export into Ultralytics YOLO format.

    Writes:
        output_dir/images/<split>/<file>
        output_dir/labels/<split>/<file>.txt   (class x_center y_center width height, normalized)
        output_dir/data.yaml                   ({path, train, val, test, nc, names})

    Unlike JumpStart's TF Object Detection API, YOLO class ids are 0-indexed with no
    reserved background class. Ultralytics requires a "val" split to train, so "valid"
    is used for it when present, otherwise it falls back to "train". Returns
    (output_dir, class_names).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)  # start clean so stale renamed images don't linger

    rows_by_split = {}
    for split in splits:
        split_dir = data_dir / split
        if split_dir.is_dir():
            rows_by_split[split] = read_annotations(split_dir)

    if not rows_by_split:
        raise ValueError(f"Found none of the splits {splits} inside {data_dir}.")

    class_names = collect_class_names(rows_by_split)
    class_to_id = {name: index for index, name in enumerate(class_names)}

    total_boxes = 0
    for split, rows in rows_by_split.items():
        split_dir = data_dir / split
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        boxes_by_image = OrderedDict()
        for row in rows:
            boxes_by_image.setdefault(row["filename"], []).append(row)

        for filename, boxes in boxes_by_image.items():
            shutil.copy2(split_dir / filename, images_dir / filename)

            width, height = boxes[0]["width"], boxes[0]["height"]
            lines = []
            for box in boxes:
                x_center = (box["xmin"] + box["xmax"]) / 2 / width
                y_center = (box["ymin"] + box["ymax"]) / 2 / height
                box_width = (box["xmax"] - box["xmin"]) / width
                box_height = (box["ymax"] - box["ymin"]) / height
                lines.append(
                    f"{class_to_id[box['class']]} {x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}"
                )
            (labels_dir / f"{Path(filename).stem}.txt").write_text("\n".join(lines) + "\n")
            total_boxes += len(boxes)

    data_yaml = {
        "path": ".",
        "train": "images/train" if "train" in rows_by_split else "images/valid",
        "val": "images/valid" if "valid" in rows_by_split else "images/train",
        "nc": len(class_names),
        "names": class_names,
    }
    if "test" in rows_by_split:
        data_yaml["test"] = "images/test"
    (output_dir / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False))

    print(f"Prepared {total_boxes} boxes across {len(class_names)} classes -> {output_dir}")
    print("Classes:", class_names)
    return output_dir, class_names
