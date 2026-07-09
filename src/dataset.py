import csv
import json
import shutil
from collections import OrderedDict
from pathlib import Path


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


def prepare_jumpstart_dataset(data_dir="dataset", output_dir="jumpstart_data", splits=("train",)):
    """Convert a Roboflow TF CSV export into SageMaker JumpStart Object Detection format.

    Writes:
        output_dir/images/<split>__<file>.jpg
        output_dir/annotations.json   {"images": [...], "annotations": [...]}
        output_dir/class_names.txt    (category_id -> name, one per line)

    JumpStart bbox is [xmin, ymin, xmax, ymax] in absolute pixels, which matches the
    Roboflow CSV directly. Returns (output_dir, class_names).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {}
    for split in splits:
        split_dir = data_dir / split
        if split_dir.is_dir():
            rows_by_split[split] = read_annotations(split_dir)

    if not rows_by_split:
        raise ValueError(f"Found none of the splits {splits} inside {data_dir}.")

    class_names = collect_class_names(rows_by_split)
    class_to_id = {name: index for index, name in enumerate(class_names)}  # 0-indexed

    images = []
    annotations = []
    image_id = 0

    for split, rows in rows_by_split.items():
        split_dir = data_dir / split
        boxes_by_image = OrderedDict()
        for row in rows:
            boxes_by_image.setdefault(row["filename"], []).append(row)

        for filename, boxes in boxes_by_image.items():
            # Prefix with split so filenames stay unique if splits are combined.
            dst_name = f"{split}__{filename}"
            shutil.copy2(split_dir / filename, images_dir / dst_name)

            first = boxes[0]
            images.append(
                {
                    "file_name": dst_name,
                    "height": first["height"],
                    "width": first["width"],
                    "id": image_id,
                }
            )
            for box in boxes:
                annotations.append(
                    {
                        "image_id": image_id,
                        "bbox": [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                        "category_id": class_to_id[box["class"]],
                    }
                )
            image_id += 1

    (output_dir / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations})
    )
    (output_dir / "class_names.txt").write_text("\n".join(class_names) + "\n")

    print(
        f"Prepared {len(images)} images, {len(annotations)} boxes, "
        f"{len(class_names)} classes -> {output_dir}"
    )
    print("Classes:", class_names)
    return output_dir, class_names
