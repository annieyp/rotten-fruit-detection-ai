import tensorflow as tf


def load_image(image_path, image_size=(224, 224)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    return tf.expand_dims(image, axis=0)


def predict_image(model_path, image_path, class_names):
    model = tf.keras.models.load_model(model_path)
    image = load_image(image_path)
    predictions = model.predict(image)
    predicted_index = int(tf.argmax(predictions[0]).numpy())
    confidence = float(predictions[0][predicted_index])

    return class_names[predicted_index], confidence

