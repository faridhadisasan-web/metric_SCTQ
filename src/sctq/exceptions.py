class SCTQError(Exception):
    """Base exception class for SCTQ framework."""

    pass


class ConfigError(SCTQError):
    """Exception raised for configuration errors."""

    pass


class DataFormatError(SCTQError):
    """Exception raised for invalid data formats (MOT, YOLO, etc.)."""

    pass


class TrackingError(SCTQError):
    """Exception raised for tracking runtime errors."""

    pass


class MetricComputationError(SCTQError):
    """Exception raised for errors during metric computation."""

    pass


class SyntheticGenerationError(SCTQError):
    """Exception raised for synthetic benchmark generation errors."""

    pass


class CorruptionError(SCTQError):
    """Exception raised for corruption runtime errors."""

    pass


class EvaluationError(SCTQError):
    """Exception raised for evaluation pipeline errors."""

    pass
