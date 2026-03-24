from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from google.cloud import firestore

GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "ai-agent-staging-490002")
FIRESTORE_DATABASE: str = os.getenv("FIRESTORE_DATABASE", "staging")


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
        print(f"Failed to load credentials: {exc}", flush=True)
        return None


@lru_cache(maxsize=1)
def _get_credentials_cached():
    return _get_credentials()


@lru_cache(maxsize=1)
def _db() -> firestore.Client:
    creds = _get_credentials_cached()
    if creds:
        return firestore.Client(
            project=GCP_PROJECT_ID,
            database=FIRESTORE_DATABASE,
            credentials=creds,
        )
    return firestore.Client(project=GCP_PROJECT_ID, database=FIRESTORE_DATABASE)


def get_agent_config(client_id: str, agent_id: str) -> dict[str, Any]:
    """Load agent config from /clients/{client_id}/agents/{agent_id}."""
    db = _db()
    doc = (
        db.collection("clients")
        .document(client_id)
        .collection("agents")
        .document(agent_id)
        .get()
    )
    if not doc.exists:
        raise ValueError(f"No agent config for client={client_id} agent_id={agent_id}")
    return doc.to_dict()


def write_event_disposition(
    client_id: str,
    event_id: str,
    disposition: dict[str, Any],
    outcome: str,
    duration_seconds: int,
    transcript_available: bool = False,
) -> None:
    """Write final disposition and outcome to Firestore event doc."""
    db = _db()
    (
        db.collection("clients")
        .document(client_id)
        .collection("events")
        .document(event_id)
        .update(
            {
                "status": "completed",
                "disposition": disposition,
                "outcome": outcome,
                "duration_seconds": duration_seconds,
                "transcript_available": transcript_available,
                "ended_at": _now_iso(),
                "updated_at": _now_iso(),
            }
        )
    )


def update_event_status(client_id: str, event_id: str, status: str) -> None:
    db = _db()
    (
        db.collection("clients")
        .document(client_id)
        .collection("events")
        .document(event_id)
        .update({"status": status, "updated_at": _now_iso()})
    )
