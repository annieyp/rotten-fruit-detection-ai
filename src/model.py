import keras
import keras_hub


# ResNet-50 ImageNet backbone is the standard RetinaNet feature extractor.
DEFAULT_BACKBONE = "resnet_50_imagenet"


def build_model(num_classes, backbone_preset=DEFAULT_BACKBONE):
    """Build a RetinaNet detector on a pretrained backbone, ready to fine-tune."""
    image_converter = keras_hub.layers.RetinaNetImageConverter(scale=1 / 255)
    preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor(
        image_converter=image_converter,
    )

    image_encoder = keras_hub.models.Backbone.from_preset(backbone_preset)
    backbone = keras_hub.models.RetinaNetBackbone(
        image_encoder=image_encoder,
        min_level=3,
        max_level=5,
        use_p5=True,
    )

    return keras_hub.models.RetinaNetObjectDetector(
        backbone=backbone,
        num_classes=num_classes,
        preprocessor=preprocessor,
        use_prediction_head_norm=True,
    )


def load_model(model_path):
    return keras.saving.load_model(model_path)
