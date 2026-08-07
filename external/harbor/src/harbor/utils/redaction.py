"""Helpers for serializing Harbor models without persisting credentials."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


REDACTED_VALUE = "<redacted>"

# Match credential-bearing environment/config names while leaving useful
# reproduction metadata (model names, base URLs, endpoints, etc.) intact.
_SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|"
    r"PRIVATE_?KEY|ACCESS_?KEY|FALLBACK_?KEY|COOKIE|AUTHORIZATION|"
    r"AUTH_?(?:TOKEN|KEY|SECRET|JSON))(?:_|$)",
    re.IGNORECASE,
)
_AUTH_HEADER_RE = re.compile(
    r"(?im)(?P<prefix>\b(?:authorization|proxy-authorization)\s*:\s*"
    r"(?:bearer|basic)\s+)(?P<value>\S+)"
)


def is_secret_key(key: object) -> bool:
    """Return whether a mapping key conventionally contains a credential."""

    if not isinstance(key, str):
        return False
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized = normalized.replace("-", "_").replace(".", "_")
    return _SECRET_KEY_RE.search(normalized) is not None


def redact_secret_fields(value: Any) -> Any:
    """Return a recursively copied value with credential fields redacted.

    This never mutates the supplied model/dictionary, so Harbor can continue to
    use the real environment values at runtime while persisted artifacts remain
    safe to share.
    """

    if isinstance(value, dict):
        return {
            key: REDACTED_VALUE if is_secret_key(key) else redact_secret_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secret_fields(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secret_fields(item) for item in value]
    return value


def redacted_model_dump_json(model: BaseModel, *, indent: int | None = None) -> str:
    """Serialize a Pydantic model after redacting credential-bearing fields."""

    dumped = model.model_dump(mode="json")
    return redact_artifact_text(
        json.dumps(redact_secret_fields(dumped), indent=indent)
    )


def redact_artifact_text(
    value: str | None,
    *credential_sources: Mapping[str, object] | None,
) -> str:
    """Redact environment-known credentials before persisting raw text.

    This is a capture-time defense for command/stdout/stderr/traceback files.
    Callers can pass the exact environment attached to a command; the current
    process environment is always included.  Secret values are never logged.
    """

    if not value:
        return value or ""
    sources: tuple[Mapping[str, object], ...] = (
        os.environ,
        *(source for source in credential_sources if source is not None),
    )
    secrets: set[str] = set()
    for source in sources:
        for key, raw in source.items():
            if (
                is_secret_key(key)
                and isinstance(raw, str)
                and len(raw) >= 8
                and raw != REDACTED_VALUE
            ):
                secrets.add(raw)
    redacted = value
    for secret in sorted(secrets, key=len, reverse=True):
        redacted = redacted.replace(secret, REDACTED_VALUE)
    return _AUTH_HEADER_RE.sub(
        lambda match: f"{match.group('prefix')}{REDACTED_VALUE}", redacted
    )
