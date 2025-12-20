import logging
import sys
from pathlib import Path
from typing import Optional, Union


class ShortNameFormatter(logging.Formatter):
    """Custom formatter that shows shortened logger names."""

    def __init__(self, prefix: str = "", fmt: Optional[str] = None):
        self.prefix = prefix
        if fmt is None:
            fmt = "[%(levelname)s] %(shortname)s: %(message)s"
        super().__init__(fmt)

    def format(self, record):
        # Add shortname attribute
        if (
            hasattr(record, "name")
            and self.prefix
            and record.name.startswith(self.prefix + ".")
        ):
            record.shortname = record.name[len(self.prefix) + 1 :]
        elif hasattr(record, "name"):
            record.shortname = record.name
        else:
            record.shortname = "unknown"
        return super().format(record)


def setup_logging(
    package_name: str = "vsf",
    log_file: Optional[Union[str, Path]] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_format: str = "[%(levelname)s] %(shortname)s: %(message)s",
    file_mode: str = "w",
    silence_third_party: bool = True,
    third_party_level: int = logging.WARNING,
) -> logging.Logger:
    """
    Set up logging for a package.

    Args:
        package_name: Name of your package (e.g., "vsf", "myapp")
        log_file: Optional file path for logging
        console_level: Console logging level
        file_level: File logging level
        log_format: Format string for log messages
        file_mode: File mode ('a' for append, 'w' for overwrite)
        silence_third_party: Whether to reduce third-party library log noise
        third_party_level: Level to set for third-party libraries

    Returns:
        The configured logger for your package
    """

    # Get or create the package logger
    logger = logging.getLogger(package_name)
    logger.setLevel(logging.DEBUG)  # Let handlers control filtering

    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        ShortNameFormatter(prefix=package_name, fmt=log_format)
    )
    logger.addHandler(console_handler)

    # Set up file handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(
            ShortNameFormatter(prefix=package_name, fmt=log_format)
        )
        logger.addHandler(file_handler)

    # Configure third-party library logging
    if silence_third_party:
        # Common noisy libraries
        noisy_loggers = [
            "urllib3",
            "requests",
            "matplotlib",
            "PIL",
            "asyncio",
            "concurrent.futures",
            "chardet",
            "werkzeug",
        ]

        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(third_party_level)

    # Don't propagate to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str, package_name: str) -> logging.Logger:
    """
    Get a logger with the proper naming convention.

    Args:
        name: Usually __name__ from the calling module
        package_name: Your package name

    Returns:
        Logger instance
    """
    # If name already starts with package_name, use as-is
    if name.startswith(package_name):
        return logging.getLogger(name)

    # Otherwise, prefix with package name
    logger_name = f"{package_name}.{name}" if name != "__main__" else package_name
    return logging.getLogger(logger_name)


# Example usage functions
def example_basic_setup():
    """Example: Basic setup with console logging only."""
    logger = setup_logging("vsf")
    return logger


def example_file_setup():
    """Example: Setup with both console and file logging."""
    logger = setup_logging(
        package_name="vsf",
        log_file="app.log",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    return logger


def example_production_setup():
    """Example: Production-ready setup."""
    logger = setup_logging(
        package_name="vsf",
        log_file="logs/vsf.log",
        console_level=logging.WARNING,  # Less noisy console
        file_level=logging.DEBUG,  # Detailed file logs
        file_mode="a",  # Don't overwrite logs
        silence_third_party=True,  # Reduce noise from dependencies
    )
    return logger


# Usage example
if __name__ == "__main__":
    # Set up logging for the 'vsf' package
    main_logger = setup_logging("vsf", log_file="example.log")

    # In your modules, use:
    logger = get_logger(__name__, "vsf")
    # or simply:
    # logger = logging.getLogger("vsf.your_module")

    # Test logging
    logger.info("Application started")
    logger.debug("Debug information")
    logger.warning("This is a warning")
    logger.error("This is an error")
