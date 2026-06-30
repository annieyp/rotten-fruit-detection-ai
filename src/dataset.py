from pathlib import Path
import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data(
    data_dir,
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.15,
    test_split=0.15,
    seed=42,
):
    data_dir = Path(data_dir)

    image_paths = []
    labels = []

    for item_dir in sorted(data_dir.iterdir()):
        if not item_dir.is_dir():
            continue

        for condition_dir in sorted(item_dir.iterdir()):
            if not condition_dir.is_dir():
                continue

            label = condition_dir.name

            for image_path in condition_dir.glob("*"):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(str(image_path))
                    labels.append(label)

    if len(image_paths) == 0:
        raise ValueError("No images found. Check your dataset folder path.")

    class_names = sorted(set(labels))
    label_to_index = {name: index for index, name in enumerate(class_names)}
    numeric_labels = [label_to_index[label] for label in labels]

    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, numeric_labels))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        return image, label

    dataset = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(
        buffer_size=len(image_paths),
        seed=seed,
        reshuffle_each_iteration=False,
    )

    total_size = len(image_paths)
    test_size = int(total_size * test_split)
    val_size = int(total_size * validation_split)

    test_set = dataset.take(test_size)
    val_set = dataset.skip(test_size).take(val_size)
    train_set = dataset.skip(test_size + val_size)

    train_set = train_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_set = val_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_set = test_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_set, val_set, test_set, class_names
