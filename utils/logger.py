import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger():
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger("cvoice")
    logger.setLevel(logging.DEBUG)  # Set root logger to DEBUG

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler - Changed to DEBUG level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Changed from INFO to DEBUG
    console_handler.setFormatter(formatter)

    # File handler - Also set to DEBUG for complete logging
    log_file = os.path.join(logs_dir, "cvoice.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
