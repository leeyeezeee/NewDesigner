"""Shared standard-library logger used by optional GDesigner tools."""

import logging


logger = logging.getLogger("GDesigner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
