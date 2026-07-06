import csv
import json
from collections import Counter, OrderedDict
from pathlib import Path


SPLITS = ("train", "valid", "test")


def get_split_dir(data_dir, split_name):
    for item in Path(data_dir).iterdir():
        if item.is_dir() and item.name.lower() == split_name:
            return item

    return None


def get_annotations_csv(split_dir):
    for name in ("_annotations.csv", "annotations.csv"):
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
        split_dir / "images" / filename,
        split_dir / filename,
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


def prepare_detection_dataset(data_dir, output_dir=None):
    dataset = collect_detection_dataset(data_dir)

    if output_dir is not None:
        summary = write_detection_metadata(dataset, output_dir)
    else:
        summary = summarize_detection_dataset(dataset)

    return dataset, summary
