"""File + console logging for the whole `src` package. Call setup_logging() from entry points."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

DEFAULT_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "app.log"
_MARK = "_rag_handler"


def setup_logging(
    log_file: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
) -> Path:
    """Attach a rotating file handler and a console handler to the `src` logger.
    Safe to call repeatedly: handlers from an earlier call are replaced, not duplicated."""
    log_file = Path(log_file) if log_file else DEFAULT_LOG_FILE
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("src")
    logger.setLevel(min(console_level, file_level))
    for handler in [h for h in logger.handlers if getattr(h, _MARK, False)]:
        logger.removeHandler(handler)
        handler.close()

    file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    for handler in (file_handler, console):
        setattr(handler, _MARK, True)
        logger.addHandler(handler)
    return log_file
