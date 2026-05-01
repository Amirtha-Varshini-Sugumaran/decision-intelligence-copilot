"""Logging helpers."""

import logging


def configure_logging() -> None:
    """Configure readable application logging once at startup."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

