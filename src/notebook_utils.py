import shutil
import zipfile
from pathlib import Path


def split_s3_uri(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")

    bucket_name, _, key = s3_uri[5:].partition("/")
    return bucket_name, key


def download_s3_file(s3_uri, local_path):
    import boto3

    bucket_name, key = split_s3_uri(s3_uri)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket_name, key, str(local_path))
    return local_path


def has_train_test(path):
    child_names = {child.name.lower() for child in Path(path).iterdir() if child.is_dir()}
    return "train" in child_names and "test" in child_names


def find_dataset_root(search_dir):
    search_dir = Path(search_dir)

    if has_train_test(search_dir):
        return search_dir

    for path in search_dir.rglob("*"):
        if path.is_dir() and has_train_test(path):
            return path

    raise ValueError(f"Could not find a folder with train and test inside {search_dir}.")


def extract_dataset_zip(zip_path, extract_dir):
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)

    if not zip_path.exists():
        raise FileNotFoundError(f"Could not find dataset zip: {zip_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    return find_dataset_root(extract_dir)


def copy_and_extract_dataset_zip(source_zip_path, local_zip_path, extract_dir):
    source_zip_path = Path(source_zip_path)
    local_zip_path = Path(local_zip_path)

    if not source_zip_path.exists():
        raise FileNotFoundError(f"Could not find dataset zip: {source_zip_path}")

    local_zip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_zip_path, local_zip_path)
    return extract_dataset_zip(local_zip_path, extract_dir)
