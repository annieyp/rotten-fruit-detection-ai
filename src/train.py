import tensorflow as tf

from dataset import load_data
from model import build_model


def train(model, train_set, val_set, epochs=10, learning_rate=0.001, weight_decay=0.0001):
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=epochs,
    )

    return history


def test_model(model, test_set):
    results = model.evaluate(test_set, return_dict=True)

    print("Test results:")
    for metric_name, value in results.items():
        print(f"{metric_name}: {value}")

    return results


def main():
    data_dir = "Unified_Dataset"

    train_set, val_set, test_set, class_names = load_data(
        data_dir=data_dir,
        image_size=(224, 224),
        batch_size=32,
    )

    print("Classes:", class_names)

    model = build_model(num_classes=len(class_names))

    train(
        model=model,
        train_set=train_set,
        val_set=val_set,
        epochs=10,
        learning_rate=0.001,
        weight_decay=0.0001,
    )

    test_model(model, test_set)

    model.save("fresh_vs_rotten_model.keras")


if __name__ == "__main__":
    main()
