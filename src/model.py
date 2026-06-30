import tensorflow as tf


def build_model(num_classes):
    base_model = tf.keras.applications.EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        include_preprocessing=True,
        input_shape=(224, 224, 3),
    )

    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=base_model.input, outputs=output)
