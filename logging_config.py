import logging
import logging.handlers
from pathlib import Path

from config import settings


LOG_FORMAT = (
    "|%(asctime)s | %(levelname)s | "
    "%(filename)s:%(funcName)s:%(lineno)d| - %(message)s"
)


class SuppressNoisyLogs(logging.Filter):
    """Suppress known noisy third-party log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()

        noisy_messages = (
            "HTTP Request:",
            "Negotiated protocol version:",
        )

        return not any(
            message.startswith(noisy_message)
            for noisy_message in noisy_messages
        )


def setup_logging() -> None:

    log_level = getattr(
        logging,
        settings.LOG_LEVEL.upper(),
        logging.INFO,
    )

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    # --------------------------------------------------------------
    # Console
    # --------------------------------------------------------------

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SuppressNoisyLogs())

    handlers.append(console_handler)

    # --------------------------------------------------------------
    # Optional file logging
    # --------------------------------------------------------------

    if settings.LOG_TO_FILE:

        log_file = Path(settings.LOG_FILE)

        log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )

        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SuppressNoisyLogs())

        handlers.append(file_handler)

    # --------------------------------------------------------------
    # Root logger
    # --------------------------------------------------------------

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # --------------------------------------------------------------
    # Third-party logger levels
    # --------------------------------------------------------------

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialized | level=%s | file_logging=%s",
        logging.getLevelName(log_level),
        settings.LOG_TO_FILE,
    )