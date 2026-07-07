import keras

from dataset import load_data
from model import build_model


def train(model, train_ds, val_ds, epochs=20, learning_rate=0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        box_loss="smooth_l1",
        classification_loss="focal",
    )
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs)


def test_model(model, test_ds):
    results = model.evaluate(test_ds, return_dict=True)
    print("Test results:", results)
    return results


def main(
    data_dir="dataset",
    model_path="retinanet_fruit.keras",
    image_size=640,
    batch_size=4,
    epochs=20,
    learning_rate=0.001,
):
    datasets, class_names = load_data(data_dir, image_size=image_size, batch_size=batch_size)

    model = build_model(num_classes=len(class_names))

    val_ds = datasets.get("valid", datasets.get("test"))
    train(model, datasets["train"], val_ds, epochs=epochs, learning_rate=learning_rate)

    if "test" in datasets:
        test_model(model, datasets["test"])

    model.save(model_path)
    print("Saved model:", model_path)
    return model


if __name__ == "__main__":
    main()
