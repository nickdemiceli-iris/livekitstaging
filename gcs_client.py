from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from google.cloud import storage

GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "ai-agent-staging-490002")
TRANSCRIPTS_BUCKET: str = os.getenv("TRANSCRIPTS_BUCKET", "iris-agent-transcripts")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_credentials():
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
    if not creds_json:
        return None
    try:
        from google.oauth2 import service_account

        try:
            decoded = base64.b64decode(creds_json).decode()
            creds_info = json.loads(decoded)
        except Exception:
            creds_info = json.loads(creds_json)
        return service_account.Credentials.from_service_account_info(creds_info)
    except Exception as exc:
        print(f"Failed to load GCS credentials: {exc}", flush=True)
        return None


@lru_cache(maxsize=1)
def _storage_client() -> storage.Client:
    creds = _get_credentials()
    if creds:
        return storage.Client(project=GCP_PROJECT_ID, credentials=creds)
    return storage.Client(project=GCP_PROJECT_ID)


def _serialize_turn(turn: Any) -> dict[str, Any]:
    if is_dataclass(turn):
        return asdict(turn)
    if isinstance(turn, dict):
        return turn
    return {
        "timestamp_utc": str(getattr(turn, "timestamp_utc", _now_iso())),
        "role": str(getattr(turn, "role", "unknown")),
        "text": str(getattr(turn, "text", "")),
        "interrupted": bool(getattr(turn, "interrupted", False)),
    }


def write_transcript(
    client_id: str,
    event_id: str,
    transcript: list[Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write transcript JSON to GCS and return gs:// path."""
    if not TRANSCRIPTS_BUCKET:
        raise ValueError("TRANSCRIPTS_BUCKET is not set")

    payload = {
        "client_id": client_id,
        "event_id": event_id,
        "created_at": _now_iso(),
        "metadata": metadata or {},
        "turns": [_serialize_turn(t) for t in transcript],
    }

    blob_path = f"clients/{client_id}/events/{event_id}/transcript.json"
    client = _storage_client()
    bucket = client.bucket(TRANSCRIPTS_BUCKET)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(json.dumps(payload, ensure_ascii=True), content_type="application/json")
    return f"gs://{TRANSCRIPTS_BUCKET}/{blob_path}"
