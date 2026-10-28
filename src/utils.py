"""Logging + retry helpers shared across modules."""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_FILE = "bot.log"

REQUIRED_ENV_VARS = (
    "POLY_API_KEY",
    "POLY_API_SECRET",
    "POLY_API_PASSPHRASE",
    "POLY_PROXY_ADDRESS",
    "POLY_PRIVATE_KEY",
    "POLYGON_RPC_URL",
)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(_LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


def require_env_vars(names: tuple[str, ...] = REQUIRED_ENV_VARS) -> None:
    """Raise a clear error at startup if any required env var is missing or blank.

    Call this once, right after `load_dotenv()`, before building the CLOB client.
    """
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            f"{', '.join(missing)}. Copy .env.example to .env and fill them in "
            "(see docs/polymarket/builders/api-keys.md for how to generate API credentials)."
        )


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    factor: float = 2.0,
    logger: logging.Logger | None = None,
) -> T:
    """Call `fn` with exponential back-off on exception. Re-raises the last exception."""
    log = logger or get_logger(__name__)
    delay = base_delay
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            log.warning(
                f"Attempt {attempt}/{attempts} failed: {exc!r}; retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            delay *= factor
    assert last_exc is not None
    raise last_exc
