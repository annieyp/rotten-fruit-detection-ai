class DetectionModelNotConfiguredError(NotImplementedError):
    pass


def build_model(*args, **kwargs):
    raise DetectionModelNotConfiguredError(
        "No object detection model has been selected yet. "
        "Choose a detector/framework first, then implement build_model for it."
    )


def load_model(*args, **kwargs):
    raise DetectionModelNotConfiguredError(
        "No object detection model has been selected yet. "
        "Choose a detector/framework first, then implement load_model for it."
    )
