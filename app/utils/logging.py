import logging
import os

log_dir = "/ScribaLLM/logs"

class ExcludeEventFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        ignore_list = [
            "InotifyEvent",
            "STREAM",
            "matplotlib"
        ]
        return not any(ignore in message for ignore in ignore_list)

def setup_logging(level=logging.INFO, console_logging=False):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "main.log")

    logger = logging.getLogger()
    logger.setLevel(level)

    logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.addFilter(ExcludeEventFilter())

    # Attach handler to root logger
    logger.addHandler(file_handler)

    if console_logging:
        # Create stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        stream_handler.addFilter(ExcludeEventFilter())  # Apply same filter
        logger.addHandler(stream_handler)
