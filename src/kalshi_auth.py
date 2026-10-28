"""Kalshi request signing (RSA-PSS over SHA-256).

Every authenticated Kalshi request — REST *and* WebSocket — carries three
headers. The signature covers `timestamp + METHOD + path`, where `path` starts
at the API root and excludes any query string.

See docs/kalshi/_llms-full.txt:7427 for the reference implementation.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.utils import get_logger

logger = get_logger(__name__)

WS_PATH = "/trade-api/ws/v2"


@dataclass(frozen=True)
class KalshiCreds:
    """One environment's credentials. Demo and production keys are not
    interchangeable — a demo key returns 401 against production and vice versa.
    """

    key_id: str
    private_key_path: str
    label: str


def load_creds(*, prod: bool = False) -> KalshiCreds:
    """Read credentials for demo (default) or production from the environment."""
    prefix = "KALSHI_PROD_" if prod else "KALSHI_"
    label = "production" if prod else "demo"

    key_id = os.getenv(f"{prefix}API_KEY_ID", "")
    key_path = os.getenv(f"{prefix}PRIVATE_KEY_PATH", "")

    if not key_id:
        raise RuntimeError(
            f"{prefix}API_KEY_ID is not set — needed for the {label} environment. "
            "It's the UUID shown when you created the API key."
        )
    if not key_path:
        raise RuntimeError(
            f"{prefix}PRIVATE_KEY_PATH is not set — needed for the {label} environment."
        )
    return KalshiCreds(key_id=key_id, private_key_path=key_path, label=label)


@lru_cache(maxsize=4)
def load_private_key(path: str | None = None) -> rsa.RSAPrivateKey:
    """Load the RSA private key Kalshi issued when the API key was created.

    Never log the key or its contents — only the path it came from.
    """
    key_path = path or os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    if not key_path:
        raise RuntimeError(
            "KALSHI_PRIVATE_KEY_PATH is not set. Point it at the .key/.pem file "
            "Kalshi gave you when you created the API key."
        )

    resolved = Path(key_path).expanduser()
    if not resolved.is_file():
        raise RuntimeError(f"Kalshi private key not found at {resolved}")

    key = serialization.load_pem_private_key(resolved.read_bytes(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise RuntimeError(f"{resolved} is not an RSA private key")

    logger.info(f"Loaded Kalshi private key from {resolved}")
    return key


def sign(method: str, path: str, *, private_key: rsa.RSAPrivateKey | None = None) -> tuple[str, str]:
    """Return `(timestamp_ms, base64_signature)` for a request to `path`."""
    key = private_key or load_private_key()
    timestamp_ms = str(int(time.time() * 1000))
    # Query strings are excluded from the signed message.
    message = f"{timestamp_ms}{method.upper()}{path.split('?')[0]}".encode()

    signature = key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return timestamp_ms, base64.b64encode(signature).decode()


def auth_headers(
    method: str,
    path: str,
    creds: KalshiCreds | None = None,
) -> dict[str, str]:
    """Build the three `KALSHI-ACCESS-*` headers for a request to `path`."""
    creds = creds or load_creds()
    key = load_private_key(creds.private_key_path)
    timestamp_ms, signature = sign(method, path, private_key=key)
    return {
        "KALSHI-ACCESS-KEY": creds.key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
    }


def ws_auth_headers(creds: KalshiCreds | None = None) -> dict[str, str]:
    """Headers for the WebSocket handshake. The signed path is the WS route."""
    return auth_headers("GET", WS_PATH, creds)
