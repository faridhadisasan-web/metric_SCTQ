import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "sctq", level: int = logging.INFO, log_file: Path | str | None = None
) -> logging.Logger:
    """Set up and return a customized logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times if logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()
