from .image_corruptions import DetectionCorruptions, ImageCorruptions


class CorruptionRegistry:
    """Registry for looking up corruption functions by name."""

    _IMAGE_CORRUPTIONS = {
        "gaussian_noise": ImageCorruptions.gaussian_noise,
        "salt_and_pepper": ImageCorruptions.salt_and_pepper,
        "gaussian_blur": ImageCorruptions.gaussian_blur,
    }

    _DETECTION_CORRUPTIONS = {
        "gaussian_jitter_center": DetectionCorruptions.gaussian_jitter_center,
        "random_drop": DetectionCorruptions.random_drop,
        "false_positives": DetectionCorruptions.false_positives,
    }

    @classmethod
    def get_image_corruption(cls, name: str):
        return cls._IMAGE_CORRUPTIONS.get(name)

    @classmethod
    def get_detection_corruption(cls, name: str):
        return cls._DETECTION_CORRUPTIONS.get(name)

    @classmethod
    def list_image_corruptions(cls):
        return list(cls._IMAGE_CORRUPTIONS.keys())

    @classmethod
    def list_detection_corruptions(cls):
        return list(cls._DETECTION_CORRUPTIONS.keys())
