from model import load_model


def predict_image(model_path, image_path, *args, **kwargs):
    model = load_model(model_path)
    return model.predict(image_path, *args, **kwargs)
