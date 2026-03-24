from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from urllib import request

from dotenv import load_dotenv
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

from knowledge_base import KnowledgeBase
from prompting import build_system_prompt

load_dotenv()


SUPPORT_CONFIG: dict[str, str] = {
    "company_name": os.getenv("SUPPORT_COMPANY_NAME", "Simple Loans"),
    "agent_name": os.getenv("SUPPORT_AGENT_NAME", "Abby"),
    "company_description": os.getenv(
        "SUPPORT_COMPANY_DESCRIPTION",
        "Simple Loans provides title loan support and customer assistance.",
    ),
}


@dataclass
class SupportDispositionState:
    issue_summary: str | None = None
    resolved: bool | None = None
    callback_offered: bool = False
    callback_requested: bool = False
    callback_phone_number: str | None = None
    callback_time_window: str | None = None
    escalation_reason: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class TranscriptTurn:
    timestamp_utc: str
    role: str
    text: str
    interrupted: bool = False


def _to_role_string(role: Any) -> str:
    value = getattr(role, "value", role)
    return str(value).lower().strip()


def _extract_item_text(item: Any) -> str:
    text_content = getattr(item, "text_content", None)
    if text_content:
        return str(text_content).strip()

    content = getattr(item, "content", None)
    if not content:
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str) and part.strip():
            parts.append(part.strip())
            continue
        transcript = getattr(part, "transcript", None)
        if transcript and str(transcript).strip():
            parts.append(str(transcript).strip())
    return " ".join(parts).strip()


def _is_user_role(role: str) -> bool:
    return role in {"user", "human", "customer"}


def _normalize_phone(phone_number: str) -> str:
    raw = phone_number.strip()
    if not raw:
        return ""
    if raw.startswith("+"):
        return raw
    digits = re.sub(r"\D+", "", raw)
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return raw


def _infer_resolution_from_text(text: str) -> bool | None:
    lowered = text.lower()
    positive = [
        "that helps",
        "yes that helps",
        "got it",
        "makes sense",
        "thank you",
        "thanks that helps",
    ]
    negative = [
        "doesn't help",
        "does not help",
        "not helpful",
        "still confused",
        "still not clear",
        "no that doesn't help",
        "no that does not help",
    ]
    if any(marker in lowered for marker in positive):
        return True
    if any(marker in lowered for marker in negative):
        return False
    return None


def _build_report(
    *,
    disposition: SupportDispositionState,
    transcript: list[TranscriptTurn],
    started_at: datetime,
    ended_at: datetime,
    room_name: str,
    trigger: str,
) -> dict[str, Any]:
    return {
        "call_metadata": {
            "room_name": room_name,
            "company_name": SUPPORT_CONFIG["company_name"],
            "agent_name": SUPPORT_CONFIG["agent_name"],
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "duration_seconds": max(0, int((ended_at - started_at).total_seconds())),
            "finalization_trigger": trigger,
        },
        "disposition": asdict(disposition),
        "transcript": [asdict(turn) for turn in transcript],
    }


def _persist_report(report: dict[str, Any]) -> str:
    output_dir = os.getenv("SUPPORT_REPORT_OUTPUT_DIR", "post_call_reports")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"{ts}_support_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return os.path.abspath(path)


def _post_report(report: dict[str, Any]) -> tuple[bool, str]:
    webhook_url = os.getenv("SUPPORT_REPORT_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False, "SUPPORT_REPORT_WEBHOOK_URL not set; skipped webhook post."

    payload = json.dumps(report).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as resp:
            return True, f"Webhook post succeeded with status {resp.status}."
    except Exception as exc:  # pragma: no cover
        return False, f"Webhook post failed: {exc}"


@lru_cache(maxsize=4)
def _load_knowledge_base(knowledge_dir: str) -> KnowledgeBase:
    kb = KnowledgeBase(knowledge_dir=knowledge_dir)
    kb.load()
    return kb


class SupportAgent(Agent):
    def __init__(
        self,
        *,
        instructions: str,
        knowledge_base: KnowledgeBase,
        disposition: SupportDispositionState,
    ) -> None:
        super().__init__(instructions=instructions)
        self._knowledge_base = knowledge_base
        self._disposition = disposition

    @function_tool()
    async def lookup_knowledge_base(
        self,
        context: RunContext,
        question: str,
        top_k: int = 3,
    ) -> str:
        """Look up policy and support facts from the local knowledge base."""
        safe_top_k = max(1, min(top_k, 5))
        return self._knowledge_base.render_tool_payload(question, top_k=safe_top_k)

    @function_tool()
    async def record_issue_details(
        self,
        context: RunContext,
        issue_summary: str,
        note: str = "",
    ) -> str:
        """Record the customer's support issue."""
        self._disposition.issue_summary = issue_summary.strip()
        if note.strip():
            self._disposition.notes.append(note.strip())
        return "Issue details recorded."

    @function_tool()
    async def record_resolution(
        self,
        context: RunContext,
        resolved: bool,
        note: str = "",
    ) -> str:
        """Record whether the caller said the answer helped."""
        self._disposition.resolved = resolved
        if note.strip():
            self._disposition.notes.append(note.strip())
        return "Resolution recorded."

    @function_tool()
    async def mark_escalation(
        self,
        context: RunContext,
        reason: str,
    ) -> str:
        """Record escalation need for human callback."""
        self._disposition.callback_offered = True
        self._disposition.escalation_reason = reason.strip() or "Escalation requested."
        return "Escalation recorded."

    @function_tool()
    async def record_callback_request(
        self,
        context: RunContext,
        callback_phone_number: str = "",
        callback_time_window: str = "",
        note: str = "",
    ) -> str:
        """Record callback request details after escalation offer."""
        self._disposition.callback_requested = True
        if callback_phone_number.strip():
            self._disposition.callback_phone_number = _normalize_phone(
                callback_phone_number
            )
        if callback_time_window.strip():
            self._disposition.callback_time_window = callback_time_window.strip()
        if note.strip():
            self._disposition.notes.append(note.strip())
        return "Callback request recorded."


def build_agent() -> tuple[SupportAgent, SupportDispositionState, KnowledgeBase]:
    company_name = SUPPORT_CONFIG["company_name"]
    agent_name = SUPPORT_CONFIG["agent_name"]
    kb_dir = os.path.abspath(os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base"))
    kb = _load_knowledge_base(kb_dir)

    instructions = build_system_prompt(
        company_name=company_name,
        agent_name=agent_name,
        company_description=SUPPORT_CONFIG["company_description"],
        knowledge_base_summary=kb.inventory_summary(),
    )
    disposition = SupportDispositionState()
    agent = SupportAgent(
        instructions=instructions,
        knowledge_base=kb,
        disposition=disposition,
    )
    return agent, disposition, kb


async def entrypoint(ctx: JobContext) -> None:
    agent, disposition, kb = build_agent()
    await ctx.connect()

    started_at = datetime.now(tz=timezone.utc)
    transcript: list[TranscriptTurn] = []
    finalized = False

    session = AgentSession(
        stt=assemblyai.STT(model=os.getenv("ASSEMBLYAI_STT_MODEL", "universal-streaming")),
        llm=openai.LLM(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
        tts=cartesia.TTS(
            model=os.getenv("CARTESIA_TTS_MODEL", "sonic-3"),
            language=os.getenv("TTS_LANGUAGE", "en"),
        ),
        vad=silero.VAD.load(),
    )

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event: Any) -> None:
        item = getattr(event, "item", None)
        if item is None:
            return
        text = _extract_item_text(item)
        if not text:
            return
        role = _to_role_string(getattr(item, "role", "unknown"))
        transcript.append(
            TranscriptTurn(
                timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
                role=role,
                text=text,
                interrupted=bool(getattr(item, "interrupted", False)),
            )
        )
        if _is_user_role(role):
            if not disposition.issue_summary and len(text.strip()) >= 12:
                disposition.issue_summary = text.strip()
            inferred = _infer_resolution_from_text(text)
            if inferred is not None:
                disposition.resolved = inferred

    def _finalize_once(trigger: str) -> None:
        nonlocal finalized
        if finalized:
            return
        finalized = True

        ended_at = datetime.now(tz=timezone.utc)
        room_name = str(getattr(ctx.room, "name", "unknown"))
        report = _build_report(
            disposition=disposition,
            transcript=transcript,
            started_at=started_at,
            ended_at=ended_at,
            room_name=room_name,
            trigger=trigger,
        )
        path = _persist_report(report)
        posted, webhook_message = _post_report(report)

        print("=== SUPPORT REPORT ===", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        print(f"Saved report: {path}", flush=True)
        print(webhook_message, flush=True)
        if posted:
            print("Webhook delivery: success", flush=True)

    @session.on("close")
    def _on_session_close(event: Any) -> None:
        reason = str(getattr(event, "reason", "unknown"))
        _finalize_once(f"session_close:{reason}")

    print(
        "Support agent started with "
        f"{kb.source_count} knowledge sources and {kb.chunk_count} chunks.",
        flush=True,
    )
    await session.start(room=ctx.room, agent=agent)
    await session.generate_reply(
        instructions=(
            "Start with exactly this line and nothing else: "
            f"\"Thank you for calling {SUPPORT_CONFIG['company_name']} customer support, "
            "how can I help you today?\""
        )
    )

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        _finalize_once("entrypoint_finally")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
