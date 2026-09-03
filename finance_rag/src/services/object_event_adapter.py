"""OSS object event normalization and idempotency primitives."""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

from finance_rag.src.core.config import OBJECT_EVENT_PREFIX, OBJECT_EVENT_SECRET


@dataclass(frozen=True)
class ObjectEvent:
    event_id: str
    event_type: str
    bucket: str
    key: str
    etag: str = ""
    version_id: str = ""


def parse_object_event(payload: dict[str, Any]) -> ObjectEvent:
    """Accept common OSS envelopes and normalize their object fields."""
    records = payload.get("events") or payload.get("Records") or [payload]
    record = records[0] if isinstance(records, list) and records else payload
    obj = record.get("oss", {}).get("object", {}) or record.get("object", {})
    bucket_data = record.get("oss", {}).get("bucket", {}) or record.get("bucket", {})
    key = str(obj.get("key") or record.get("key") or "")
    return ObjectEvent(
        event_id=str(record.get("eventId") or record.get("event_id") or hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()),
        event_type=str(record.get("eventName") or record.get("event_type") or "ObjectCreated"),
        bucket=str(bucket_data.get("name") or record.get("bucket_name") or ""),
        key=key,
        etag=str(obj.get("eTag") or obj.get("etag") or record.get("etag") or "").strip('"'),
        version_id=str(obj.get("versionId") or obj.get("version_id") or ""),
    )


def verify_signature(raw_body: bytes, signature: str | None) -> bool:
    if not OBJECT_EVENT_SECRET:
        return True
    expected = hmac.new(OBJECT_EVENT_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return bool(signature) and hmac.compare_digest(expected, signature)


def accepts_key(key: str) -> bool:
    return key.startswith(OBJECT_EVENT_PREFIX) and not key.startswith(f"{OBJECT_EVENT_PREFIX}_internal/")


__all__ = ["ObjectEvent", "accepts_key", "parse_object_event", "verify_signature"]
