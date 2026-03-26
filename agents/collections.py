from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any

import google.auth.transport.requests
import google.oauth2.id_token
import requests
from google.cloud import firestore, storage
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import assemblyai, cartesia, openai, silero

from prompting import build_system_prompt, get_delinquency_policy


PROJECT_ID = os.environ.get("PROJECT_ID", "").strip()
DEFAULT_COMPANY_NAME = os.environ.get("COMPANY_NAME", "IRIS Insights").strip()
WORKER_AGENT_NAME = os.environ.get("WORKER_AGENT_NAME", "collections-agent").strip()
BUCKET_NAME = os.environ.get("BUCKET_NAME", "").strip()
POST_CALL_SERVICE_URL = os.environ.get("POST_CALL_SERVICE_URL", "").strip().rstrip("/")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.irisinsights.ai/v2").strip().rstrip("/")
GCP_SERVICE_ACCOUNT_JSON = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "").strip()
GCP_SERVICE_ACCOUNT_JSON_B64 = os.environ.get("GCP_SERVICE_ACCOUNT_JSON_B64", "").strip()


@dataclass
class CallDispositionState:
    full_payment_committed: bool = False
    partial_payment_committed: bool = False
    partial_payment_amount_usd: float | None = None
    remaining_balance_commitment_date: str | None = None
    payment_plan_proposed: bool = False
    payment_plan_terms: str | None = None
    human_agent_handoff_requested: bool = False
    notes: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _configure_google_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    raw_json = GCP_SERVICE_ACCOUNT_JSON
    if not raw_json and GCP_SERVICE_ACCOUNT_JSON_B64:
        raw_json = base64.b64decode(GCP_SERVICE_ACCOUNT_JSON_B64).decode("utf-8")
    if not raw_json:
        return
    fd, path = tempfile.mkstemp(prefix="gcp-sa-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(raw_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


def _build_customer_context(customer: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- Company: {DEFAULT_COMPANY_NAME}",
            f"- Customer Name: {customer.get('customer_name', 'Unknown')}",
            (
                "- Vehicle: "
                f"{customer.get('vehicle_year', 'N/A')} "
                f"{customer.get('vehicle_make', 'N/A')} "
                f"{customer.get('vehicle_model', 'N/A')}"
            ),
            f"- Loan Origination Date: {customer.get('loan_origination_date', 'N/A')}",
            f"- Original Payment Due Date: {customer.get('original_payment_due_date', 'N/A')}",
            f"- Amount Outstanding: ${_safe_float(customer.get('amount_outstanding_usd'), 0.0):.2f}",
            f"- Days Past Due: {_safe_int(customer.get('days_past_due'), 0)}",
            f"- Loan ID: {customer.get('loan_reference_id', 'N/A')}",
            f"- Phone Number: {customer.get('phone_number', 'N/A')}",
            f"- Contact Preference: {customer.get('contact_channel_preference', 'phone')}",
            f"- Language Preference: {customer.get('language_preference', 'English')}",
            f"- Today's Date: {date.today().isoformat()}",
        ]
    )


def _load_call_metadata(raw_metadata: str | None) -> dict[str, Any]:
    if not raw_metadata:
        return {}
    try:
        data = json.loads(raw_metadata)
        if isinstance(data, dict):
            return data
        return {}
    except json.JSONDecodeError:
        return {}


def _event_doc_ref(
    db: firestore.Client, client_id: str, event_id: str
) -> firestore.DocumentReference:
    return db.collection("clients").document(client_id).collection("events").document(event_id)


def _base_event_fields(metadata: dict[str, Any], customer: dict[str, Any]) -> dict[str, Any]:
    event_id = str(metadata.get("event_id") or "")
    workflow = str(metadata.get("workflow") or "collections")
    return {
        "id": event_id,
        "api_version": "2.0",
        "event_schema_version": "1.0",
        "workflow": workflow,
        "client_id": metadata.get("client_id") or metadata.get("tenant_id"),
        "tenant_id": metadata.get("tenant_id"),
        "call_type": workflow,
        "handled_by": "ai",
        "agent_name": metadata.get("dispatch_agent_name") or WORKER_AGENT_NAME,
        "agent_system": "livekit",
        "agent_version": os.environ.get("AGENT_VERSION", "v1"),
        "workflow_version": f"{workflow}-v1",
        "status": "active",
        "customer_name": customer.get("customer_name"),
        "phone_number": customer.get("phone_number"),
        "loan_reference_id": customer.get("loan_reference_id"),
        "days_past_due": _safe_int(customer.get("days_past_due"), 0),
        "amount_outstanding_usd": _safe_float(customer.get("amount_outstanding_usd"), 0.0),
        "recording": {
            "status": "pending",
            "url": f"{API_BASE_URL}/events/{event_id}/recording",
            "path_hint": metadata.get("recording_gcs_path_hint"),
        },
        "transcript": {
            "status": "pending",
            "url": f"{API_BASE_URL}/events/{event_id}/transcript",
            "schema_version": "1.0",
        },
        "updated_at": _now_iso(),
    }


def _extract_turns_from_session_report(session_report: Any) -> list[dict[str, Any]]:
    report_dict: dict[str, Any] = {}
    if session_report is None:
        return []

    if hasattr(session_report, "to_dict"):
        report_dict = session_report.to_dict()  # type: ignore[assignment]
    elif hasattr(session_report, "model_dump"):
        report_dict = session_report.model_dump()  # type: ignore[assignment]
    elif isinstance(session_report, dict):
        report_dict = session_report

    possible_keys = [
        "conversation",
        "conversation_history",
        "history",
        "messages",
        "chat_history",
    ]
    rows: list[dict[str, Any]] = []
    for key in possible_keys:
        value = report_dict.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.append(
                        {
                            "role": str(item.get("role") or item.get("speaker") or "unknown"),
                            "text": str(item.get("text") or item.get("content") or "").strip(),
                            "timestamp": item.get("timestamp") or item.get("created_at"),
                        }
                    )
            if rows:
                return [row for row in rows if row["text"]]

    if report_dict:
        return [
            {
                "role": "system",
                "text": json.dumps(report_dict, ensure_ascii=False),
                "timestamp": _now_iso(),
            }
        ]

    return []


def _build_transcript_csv(turns: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["index", "speaker", "transcript", "timestamp"])
    writer.writeheader()
    for index, turn in enumerate(turns):
        writer.writerow(
            {
                "index": index,
                "speaker": turn.get("role", "unknown"),
                "transcript": turn.get("text", ""),
                "timestamp": turn.get("timestamp") or "",
            }
        )
    return output.getvalue()


def _store_transcript_artifacts(
    *,
    storage_client: storage.Client,
    bucket_name: str,
    client_id: str,
    event_id: str,
    turns: list[dict[str, Any]],
) -> dict[str, str]:
    bucket = storage_client.bucket(bucket_name)
    base_path = f"clients/{client_id}/events/{event_id}"
    json_path = f"{base_path}/transcript.json"
    csv_path = f"{base_path}/transcript.csv"

    transcript_json = {
        "transcript_schema_version": "1.0",
        "event_id": event_id,
        "generated_at": _now_iso(),
        "turns": turns,
    }
    bucket.blob(json_path).upload_from_string(
        data=json.dumps(transcript_json, ensure_ascii=False),
        content_type="application/json",
    )

    transcript_csv = _build_transcript_csv(turns)
    bucket.blob(csv_path).upload_from_string(
        data=transcript_csv,
        content_type="text/csv",
    )

    return {"json_path": json_path, "csv_path": csv_path}


def _publish_post_call(client_id: str, event_id: str) -> None:
    if not POST_CALL_SERVICE_URL:
        return
    auth_request = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_request, POST_CALL_SERVICE_URL)
    response = requests.post(
        POST_CALL_SERVICE_URL,
        json={"client_id": client_id, "event_id": event_id},
        headers={"Authorization": f"Bearer {id_token}"},
        timeout=15,
    )
    response.raise_for_status()


async def _attempt_end_call(session: AgentSession, ctx: JobContext) -> None:
    for method_name in ("aclose", "close", "end"):
        method = getattr(session, method_name, None)
        if callable(method):
            result = method()
            if asyncio.iscoroutine(result):
                await result
            break

    room = getattr(ctx, "room", None)
    if room is not None:
        for method_name in ("disconnect", "close"):
            method = getattr(room, method_name, None)
            if callable(method):
                result = method()
                if asyncio.iscoroutine(result):
                    await result
                break


class CollectionsAgent(Agent):
    def __init__(self, instructions: str, disposition: CallDispositionState) -> None:
        super().__init__(instructions=instructions)
        self._disposition = disposition

    @function_tool
    async def mark_full_payment_committed(
        self,
        context: RunContext,
        note: str = "",
    ) -> str:
        """Record that customer commits to full payment now."""
        self._disposition.full_payment_committed = True
        self._disposition.partial_payment_committed = False
        self._disposition.payment_plan_proposed = False
        if note:
            self._disposition.notes = note
        return "Full-payment commitment recorded."

    @function_tool
    async def mark_partial_payment_committed(
        self,
        context: RunContext,
        partial_payment_amount_usd: float,
        remaining_balance_commitment_date: str,
        note: str = "",
    ) -> str:
        """Record partial payment and date commitment for remaining balance."""
        self._disposition.partial_payment_committed = True
        self._disposition.partial_payment_amount_usd = partial_payment_amount_usd
        self._disposition.remaining_balance_commitment_date = (
            remaining_balance_commitment_date
        )
        if note:
            self._disposition.notes = note
        return "Partial-payment commitment recorded."

    @function_tool
    async def mark_payment_plan_proposed(
        self,
        context: RunContext,
        payment_plan_terms: str,
        note: str = "",
    ) -> str:
        """Record proposed payment plan terms accepted by the customer."""
        self._disposition.payment_plan_proposed = True
        self._disposition.payment_plan_terms = payment_plan_terms
        if note:
            self._disposition.notes = note
        return "Payment-plan terms recorded."

    @function_tool
    async def mark_human_handoff_requested(
        self,
        context: RunContext,
        reason: str = "",
    ) -> str:
        """Record customer agreement to continue with a human specialist."""
        self._disposition.human_agent_handoff_requested = True
        if reason:
            self._disposition.notes = reason
        return "Human-handoff request recorded."


def _build_agent(
    customer_profile: dict[str, Any],
) -> tuple[CollectionsAgent, CallDispositionState]:
    policy = get_delinquency_policy(_safe_int(customer_profile.get("days_past_due"), 0))
    customer_context = _build_customer_context(customer_profile)
    prompt = build_system_prompt(
        customer_context=customer_context,
        consequence_message=policy.consequence_message,
        customer_name=str(customer_profile.get("customer_name", "customer")),
    )
    disposition = CallDispositionState()
    return CollectionsAgent(instructions=prompt, disposition=disposition), disposition


async def entrypoint(ctx: JobContext) -> None:
    metadata = _load_call_metadata(getattr(ctx.job, "metadata", None))
    customer = metadata.get("customer_profile") or {}
    if not isinstance(customer, dict):
        customer = {}

    customer.setdefault("customer_name", "Customer")
    customer.setdefault("language_preference", "English")
    customer.setdefault("phone_number", metadata.get("phone_number", ""))
    customer.setdefault("days_past_due", 0)
    customer.setdefault("amount_outstanding_usd", 0.0)

    client_id = str(metadata.get("client_id") or metadata.get("tenant_id") or "").strip()
    event_id = str(metadata.get("event_id") or "").strip()

    db = firestore.Client(project=PROJECT_ID or None)
    storage_client = storage.Client(project=PROJECT_ID or None)

    event_ref = None
    if client_id and event_id:
        event_ref = _event_doc_ref(db, client_id, event_id)
        event_ref.set(_base_event_fields(metadata, customer), merge=True)

    agent, disposition = _build_agent(customer)
    await ctx.connect()

    session = AgentSession(
        stt=assemblyai.STT(model="universal-streaming-multilingual"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-3", language="en"),
        vad=silero.VAD.load(),
    )

    await session.start(room=ctx.room, agent=agent)
    await session.generate_reply(
        instructions=(
            "Start the call now by verifying whether you are speaking with the customer. "
            "Do not share any account details until identity is confirmed. "
            "If call objective is completed or voicemail action is complete, end the call deterministically. "
            "Then follow the required collections flow."
        )
    )

    session_report: Any = None
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            session_report = ctx.make_session_report()
        except Exception:
            session_report = None

        transcript_error = None
        transcript_paths: dict[str, str] = {}
        turns = _extract_turns_from_session_report(session_report)
        if BUCKET_NAME and client_id and event_id:
            try:
                transcript_paths = _store_transcript_artifacts(
                    storage_client=storage_client,
                    bucket_name=BUCKET_NAME,
                    client_id=client_id,
                    event_id=event_id,
                    turns=turns,
                )
            except Exception as exc:
                transcript_error = str(exc)

        await _attempt_end_call(session, ctx)

        if event_ref is not None:
            transcript_status = "ready" if transcript_paths else "failed"
            event_ref.set(
                {
                    "status": "completed",
                    "disposition": asdict(disposition),
                    "updated_at": _now_iso(),
                    "ended_at": _now_iso(),
                    "transcript": {
                        "status": transcript_status,
                        "url": f"{API_BASE_URL}/events/{event_id}/transcript",
                        "schema_version": "1.0",
                        "gcs_json_path": transcript_paths.get("json_path"),
                        "gcs_csv_path": transcript_paths.get("csv_path"),
                        "error": transcript_error,
                    },
                },
                merge=True,
            )
            if transcript_paths:
                _publish_post_call(client_id, event_id)

        print("=== CALL DISPOSITION SUMMARY ===")
        print(json.dumps(asdict(disposition), indent=2))


if __name__ == "__main__":
    _configure_google_credentials()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=WORKER_AGENT_NAME,
        )
    )
