from pathlib import Path

import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["fresh", "rotten"]


def get_label(folder_name):
    folder_name = folder_name.lower()

    if folder_name.startswith("fresh"):
        return "fresh"
    if folder_name.startswith("rotten"):
        return "rotten"

    raise ValueError(
        f"Could not infer label from folder '{folder_name}'. "
        "Folder names must start with 'fresh' or 'rotten'."
    )


def get_split_dir(data_dir, split_name):
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.lower() == split_name.lower():
            return item

    raise ValueError(f"Could not find '{split_name}' folder inside {data_dir}.")


def collect_images(split_dir):
    image_paths = []
    labels = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = get_label(class_dir.name)

        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(image_path))
                labels.append(label)

    return image_paths, labels


def make_dataset(image_paths, labels, label_to_index, image_size):
    numeric_labels = [label_to_index[label] for label in labels]
    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, numeric_labels))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        return image, label

    return path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)


def load_data(
    data_dir,
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.15,
    test_split=0.15,
    seed=42,
):
    data_dir = Path(data_dir)

    train_dir = get_split_dir(data_dir, "Train")
    test_dir = get_split_dir(data_dir, "Test")

    train_paths, train_labels = collect_images(train_dir)
    test_paths, test_labels = collect_images(test_dir)

    if len(train_paths) == 0:
        raise ValueError("No training images found. Check your dataset/Train folder.")
    if len(test_paths) == 0:
        raise ValueError("No test images found. Check your dataset/Test folder.")

    class_names = CLASS_NAMES
    label_to_index = {name: index for index, name in enumerate(class_names)}

    train_dataset = make_dataset(train_paths, train_labels, label_to_index, image_size)
    train_dataset = train_dataset.shuffle(
        buffer_size=len(train_paths),
        seed=seed,
        reshuffle_each_iteration=False,
    )
    test_set = make_dataset(test_paths, test_labels, label_to_index, image_size)

    val_size = int(len(train_paths) * validation_split)

    val_set = train_dataset.take(val_size)
    train_set = train_dataset.skip(val_size)

    train_set = train_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_set = val_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_set = test_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_set, val_set, test_set, class_names
