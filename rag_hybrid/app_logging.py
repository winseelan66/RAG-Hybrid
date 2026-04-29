from __future__ import annotations

import logging
from pathlib import Path

from rag_hybrid.config import get_settings

LOG_FILE_PATH = Path(__file__).resolve().parent.parent / get_settings().logging.file_name


def get_logger() -> logging.Logger:
    logger = logging.getLogger("rag_hybrid")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
