from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli, tts as lk_tts
from livekit.plugins import assemblyai, cartesia, openai, silero

try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
except Exception:  # pragma: no cover - fallback for environments missing the plugin
    MultilingualModel = None

from agents.sales import build_sales_agent, derive_outcome
from firestore_client import get_agent_config, update_event_status, write_event_disposition
from gcs_client import write_transcript

load_dotenv()

# Cartesia "Katie" is a stable, high-quality feminine voice for sales conversations.
CARTESIA_DEFAULT_FEMALE_VOICE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _env_first(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _env_float_any(names: tuple[str, ...], default: float) -> float:
    for name in names:
        value = os.getenv(name, "").strip()
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


def _env_int_any(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.getenv(name, "").strip()
        if not value:
            continue
        try:
            return int(float(value))
        except ValueError:
            continue
    return default


def _env_bool_any(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        value = os.getenv(name, "").strip().lower()
        if not value:
            continue
        return value in {"1", "true", "yes", "on"}
    return default


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return float(stripped)
        except ValueError:
            return default
    return default


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(float(stripped))
        except ValueError:
            return default
    return default


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return default
        if stripped in {"1", "true", "yes", "on"}:
            return True
        if stripped in {"0", "false", "no", "off"}:
            return False
    return default


# Lazy init so deploy-time "python main.py download-files" does not fail.
_VAD_MODEL: Any | None = None


def _get_vad_model() -> Any:
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL = silero.VAD.load()
    return _VAD_MODEL


def _build_turn_detector() -> Any | None:
    mode = _env_first(("AGENT_TURN_DETECTION", "LK_TURN_DETECTION_MODE"), "multilingual").strip().lower()
    if mode in {"off", "none", "vad", "stt"}:
        return None
    if not _env_bool_any(("LK_ENABLE_TURN_DETECTOR",), True):
        return None
    if MultilingualModel is None:
        print("Turn detector plugin unavailable; continuing without turn detector.", flush=True)
        return None
    threshold = _clamp_float(_env_float("LK_TURN_UNLIKELY_THRESHOLD", 0.25), 0.0, 1.0)
    try:
        return MultilingualModel(unlikely_threshold=threshold)
    except Exception as exc:
        # Never fail call start just because turn detector isn't available.
        print(f"Turn detector disabled at runtime: {exc}", flush=True)
        return None


def _safe_parse_metadata(raw_metadata: Any) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if not raw_metadata:
        return {}
    try:
        parsed = json.loads(str(raw_metadata))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        decoded = base64.b64decode(str(raw_metadata)).decode()
        parsed = json.loads(decoded)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


@dataclass
class TranscriptTurn:
    timestamp_utc: str
    role: str
    text: str
    interrupted: bool = False


@dataclass
class RuntimeTuning:
    stt_model: str
    stt_language: str
    turn_detection_mode: str
    use_turn_handling: bool
    tts_provider: str
    tts_model: str
    tts_voice: str
    cartesia_tts_speed: float
    openai_tts_model: str
    openai_tts_voice: str
    tts_fallback_enabled: bool
    tts_fallback_max_retry_per_provider: int
    stt_buffer_size_seconds: float
    stt_min_turn_silence_ms: int
    stt_max_turn_silence_ms: int
    stt_eot_confidence_threshold: float
    min_endpointing_delay: float
    max_endpointing_delay: float
    min_consecutive_speech_delay: float
    allow_interruptions: bool
    interruption_mode: str
    min_interruption_duration: float
    false_interruption_timeout: float | None
    resume_false_interruption: bool
    llm_temperature: float
    llm_max_completion_tokens: int
    preemptive_generation: bool


def _build_runtime_tuning(agent_config: dict[str, Any]) -> RuntimeTuning:
    low_latency_mode = _env_bool_any(("LK_LOW_LATENCY_MODE",), True)

    stt_language = _env_first(("ASSEMBLYAI__LANGUAGE", "LK_STT_LANGUAGE", "TTS_LANGUAGE"), "en").strip().lower() or "en"
    raw_turn_detection_mode = _env_first(("AGENT_TURN_DETECTION", "LK_TURN_DETECTION_MODE"), "vad").strip().lower()
    if raw_turn_detection_mode not in {"multilingual", "vad", "stt", "off", "none"}:
        raw_turn_detection_mode = "vad"
    use_turn_handling = _env_bool_any(("LK_USE_TURN_HANDLING",), True)

    # Critical reliability fix: AssemblyAI requires 50ms..1000ms.
    raw_buffer = _to_float(
        agent_config.get("stt_buffer_size_seconds"),
        _env_float_any(("LK_STT_BUFFER_SIZE_SECONDS", "ASSEMBLYAI_BUFFER_SIZE_SECONDS"), 0.05),
    )
    stt_buffer = _clamp_float(raw_buffer, 0.05, 1.0)
    if stt_buffer != raw_buffer:
        print(
            f"Adjusted stt_buffer_size_seconds from {raw_buffer} to {stt_buffer} (AssemblyAI-safe).",
            flush=True,
        )

    raw_min_turn = _to_int(
        agent_config.get("stt_min_turn_silence_ms"),
        _env_int_any(("LK_STT_MIN_TURN_SILENCE_MS", "ASSEMBLYAI_MIN_EOT_SILENCE_MS"), 160),
    )
    raw_max_turn = _to_int(
        agent_config.get("stt_max_turn_silence_ms"),
        _env_int_any(("LK_STT_MAX_TURN_SILENCE_MS", "ASSEMBLYAI_MAX_TURN_SILENCE_MS"), 550),
    )
    min_turn = _clamp_int(raw_min_turn, 150, 2000)
    max_turn = _clamp_int(raw_max_turn, 300, 5000)
    if max_turn < min_turn:
        max_turn = min_turn

    eot_conf = _clamp_float(
        _to_float(
            agent_config.get("stt_end_of_turn_confidence_threshold"),
            _env_float_any(
                ("LK_STT_END_OF_TURN_CONFIDENCE_THRESHOLD", "ASSEMBLYAI_END_OF_TURN_CONFIDENCE_THRESHOLD"),
                0.42,
            ),
        ),
        0.0,
        1.0,
    )

    min_ep = _clamp_float(
        _to_float(
            agent_config.get("min_endpointing_delay"),
            _env_float("LK_MIN_ENDPOINTING_DELAY", 0.08),
        ),
        0.0,
        5.0,
    )
    max_ep = _clamp_float(
        _to_float(
            agent_config.get("max_endpointing_delay"),
            _env_float("LK_MAX_ENDPOINTING_DELAY", 0.45),
        ),
        0.1,
        8.0,
    )
    if max_ep < min_ep:
        max_ep = min_ep

    min_consecutive = _clamp_float(
        _to_float(
            agent_config.get("min_consecutive_speech_delay"),
            _env_float("LK_MIN_CONSECUTIVE_SPEECH_DELAY", 0.05),
        ),
        0.0,
        1.5,
    )
    allow_interruptions = _to_bool(
        agent_config.get("allow_interruptions"),
        _env_bool_any(("INTERRUPT_SPEECH_ON_USER_INPUT", "LK_ALLOW_INTERRUPTIONS"), True),
    )
    interruption_mode = _env_first(("LK_INTERRUPTION_MODE",), "vad").strip().lower()
    if interruption_mode not in {"vad", "adaptive"}:
        interruption_mode = "vad"
    min_interruption_duration = _clamp_float(
        _env_float_any(("LK_MIN_INTERRUPTION_DURATION",), 0.2 if low_latency_mode else 0.35),
        0.05,
        2.0,
    )
    false_interruption_timeout_raw = _env_first(("LK_FALSE_INTERRUPTION_TIMEOUT",), "")
    false_interruption_timeout: float | None
    if false_interruption_timeout_raw.strip().lower() == "none":
        false_interruption_timeout = None
    else:
        false_interruption_timeout = _clamp_float(
            _env_float_any(("LK_FALSE_INTERRUPTION_TIMEOUT",), 1.0 if low_latency_mode else 1.5),
            0.2,
            5.0,
        )
    resume_false_interruption = _env_bool_any(("LK_RESUME_FALSE_INTERRUPTION",), False)

    llm_temp = _clamp_float(
        _to_float(agent_config.get("llm_temperature"), _env_float("LK_LLM_TEMPERATURE", 0.15)),
        0.0,
        1.2,
    )
    llm_max_tokens = _clamp_int(
        _to_int(
            agent_config.get("llm_max_completion_tokens"),
            _env_int("LK_LLM_MAX_COMPLETION_TOKENS", 64),
        ),
        32,
        600,
    )

    raw_stt_model = str(
        agent_config.get(
            "stt_model",
            _env_first(
                ("LK_STT_MODEL", "ASSEMBLYAI_MODEL", "ASSEMBLYAI__MODEL"),
                "universal-streaming-english",
            ),
        )
    ).strip()
    allowed_stt_models = {
        "universal-streaming-english",
        "universal-streaming-multilingual",
        "u3-rt-pro",
        "u3-pro",
    }
    stt_model = raw_stt_model if raw_stt_model in allowed_stt_models else "universal-streaming-english"
    if stt_model != raw_stt_model:
        print(f"Unsupported stt_model '{raw_stt_model}', using '{stt_model}'", flush=True)
    # For English-only outbound campaigns, the English model generally responds faster.
    if (
        low_latency_mode
        and stt_model == "universal-streaming-multilingual"
        and stt_language.startswith("en")
        and _env_bool_any(("LK_PREFER_ENGLISH_STT",), True)
    ):
        stt_model = "universal-streaming-english"
        print("Switching STT model to universal-streaming-english for lower latency.", flush=True)

    cartesia_enabled = _env_bool_any(("CARTESIA_TTS_ENABLED",), True)
    env_tts_provider = _env_first(("LK_TTS_PROVIDER",), "").strip().lower()
    if "tts_provider" in agent_config:
        raw_tts_provider = str(agent_config.get("tts_provider", "cartesia")).strip().lower()
    elif env_tts_provider:
        raw_tts_provider = env_tts_provider
    elif cartesia_enabled:
        # Keep Cartesia as primary when explicitly enabled.
        raw_tts_provider = "cartesia"
    else:
        raw_tts_provider = "cartesia"
    tts_provider = raw_tts_provider if raw_tts_provider in {"openai", "cartesia"} else "openai"
    if tts_provider != raw_tts_provider:
        print(f"Unsupported tts_provider '{raw_tts_provider}', using '{tts_provider}'", flush=True)

    openai_tts_model = _env_first(("LK_OPENAI_TTS_MODEL", "OPENAI_TTS_MODEL"), "gpt-4o-mini-tts")
    if openai_tts_model not in {"gpt-4o-mini-tts", "gpt-4o-realtime-preview-tts"}:
        print(f"Unsupported openai tts model '{openai_tts_model}', using 'gpt-4o-mini-tts'", flush=True)
        openai_tts_model = "gpt-4o-mini-tts"
    openai_tts_voice = _env_first(("LK_OPENAI_TTS_VOICE", "OPENAI_TTS_VOICE"), "ash")
    tts_fallback_enabled = _env_bool_any(("OPENAI_TTS_FALLBACK_ENABLED", "LK_TTS_OPENAI_FALLBACK_ENABLED"), False)
    tts_fallback_max_retry_per_provider = _clamp_int(
        _env_int_any(("TTS_FALLBACK_MAX_RETRY_PER_PROVIDER",), 1),
        1,
        5,
    )

    if tts_provider == "cartesia":
        raw_tts_model = str(
            agent_config.get(
                "tts_model",
                _env_first(("LK_TTS_MODEL", "CARTESIA_TTS_MODEL"), "sonic-3"),
            )
        ).strip()
        allowed_tts_models = {"sonic-turbo", "sonic-3", "sonic-2", "sonic"}
        tts_model = raw_tts_model if raw_tts_model in allowed_tts_models else "sonic-turbo"
        if tts_model != raw_tts_model:
            print(f"Unsupported cartesia tts_model '{raw_tts_model}', using '{tts_model}'", flush=True)
        tts_voice = str(
            agent_config.get(
                "voice",
                _env_first(("LK_CARTESIA_VOICE", "CARTESIA_TTS_VOICE"), CARTESIA_DEFAULT_FEMALE_VOICE),
            )
        ).strip()
        cartesia_tts_speed = _clamp_float(
            _to_float(
                agent_config.get("tts_speed"),
                _env_float_any(("LK_CARTESIA_TTS_SPEED", "CARTESIA_TTS_SPEED"), 1.0),
            ),
            0.85,
            1.20,
        )
    else:
        raw_tts_model = str(
            agent_config.get(
                "openai_tts_model",
                _env_first(("LK_OPENAI_TTS_MODEL", "OPENAI_TTS_MODEL"), "gpt-4o-mini-tts"),
            )
        ).strip()
        allowed_tts_models = {"gpt-4o-mini-tts", "gpt-4o-realtime-preview-tts"}
        tts_model = raw_tts_model if raw_tts_model in allowed_tts_models else "gpt-4o-mini-tts"
        if tts_model != raw_tts_model:
            print(f"Unsupported openai tts_model '{raw_tts_model}', using '{tts_model}'", flush=True)
        tts_voice = (
            str(
                agent_config.get(
                    "openai_tts_voice",
                    _env_first(("LK_OPENAI_TTS_VOICE", "OPENAI_TTS_VOICE"), "alloy"),
                )
            ).strip()
            or "alloy"
        )
        cartesia_tts_speed = 1.0

    return RuntimeTuning(
        stt_model=stt_model,
        stt_language=stt_language,
        turn_detection_mode=raw_turn_detection_mode,
        use_turn_handling=use_turn_handling,
        tts_provider=tts_provider,
        tts_model=tts_model,
        tts_voice=tts_voice,
        cartesia_tts_speed=cartesia_tts_speed,
        openai_tts_model=openai_tts_model,
        openai_tts_voice=openai_tts_voice,
        tts_fallback_enabled=tts_fallback_enabled,
        tts_fallback_max_retry_per_provider=tts_fallback_max_retry_per_provider,
        stt_buffer_size_seconds=stt_buffer,
        stt_min_turn_silence_ms=min_turn,
        stt_max_turn_silence_ms=max_turn,
        stt_eot_confidence_threshold=eot_conf,
        min_endpointing_delay=min_ep,
        max_endpointing_delay=max_ep,
        min_consecutive_speech_delay=min_consecutive,
        allow_interruptions=allow_interruptions,
        interruption_mode=interruption_mode,
        min_interruption_duration=min_interruption_duration,
        false_interruption_timeout=false_interruption_timeout,
        resume_false_interruption=resume_false_interruption,
        llm_temperature=llm_temp,
        llm_max_completion_tokens=llm_max_tokens,
        preemptive_generation=_to_bool(
            agent_config.get("preemptive_generation"),
            _env_bool_any(("LK_PREEMPTIVE_GENERATION", "AGENT_PREEMPTIVE_SYNTHESIS"), True),
        ),
    )


def _to_role_string(role: Any) -> str:
    value = getattr(role, "value", role)
    return str(value).lower()


def _extract_item_text(item: Any) -> str:
    text_content = getattr(item, "text_content", None)
    if text_content:
        return str(text_content).strip()

    content = getattr(item, "content", None)
    if not content:
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            stripped = part.strip()
            if stripped:
                parts.append(stripped)
            continue

        transcript = getattr(part, "transcript", None)
        if transcript and str(transcript).strip():
            parts.append(str(transcript).strip())

    return " ".join(parts).strip()


def _resolve_turn_detection(
    tuning: RuntimeTuning,
    model_turn_detector: Any | None,
) -> Any | str | None:
    if tuning.turn_detection_mode in {"off", "none"}:
        return None
    if tuning.turn_detection_mode == "stt":
        return "stt"
    if tuning.turn_detection_mode == "vad":
        return "vad"
    # multilingual mode: use model if available, fallback to VAD for reliability.
    if model_turn_detector is not None:
        return model_turn_detector
    return "vad"


async def entrypoint(ctx: JobContext) -> None:
    raw_metadata = getattr(ctx.job, "metadata", None) or "{}"
    metadata = _safe_parse_metadata(raw_metadata)

    client_id = str(metadata.get("client_id", "verifacto"))
    event_id = str(metadata.get("event_id", ""))
    agent_id = str(metadata.get("agent_id", "sales"))
    contact = metadata.get("contact", {}) if isinstance(metadata.get("contact"), dict) else {}

    print(f"Starting call: client={client_id} event={event_id} agent={agent_id}", flush=True)

    try:
        agent_config = get_agent_config(client_id, agent_id)
    except Exception as exc:
        print(f"Failed to load agent config: {exc}", flush=True)
        agent_config = {}
    if not isinstance(agent_config, dict):
        print("Agent config had unexpected type; using empty defaults.", flush=True)
        agent_config = {}

    agent, disposition = build_sales_agent(contact, agent_config)

    await ctx.connect()
    call_started_at = datetime.now(timezone.utc)
    transcript: list[TranscriptTurn] = []
    finalized = False

    if client_id and event_id:
        try:
            update_event_status(client_id, event_id, "active")
        except Exception as exc:
            print(f"Could not update event status: {exc}", flush=True)

    tuning = _build_runtime_tuning(agent_config)
    llm_model = str(agent_config.get("llm_model", _env_first(("OPENAI_MODEL", "OPENAI__MODEL"), "gpt-4o-mini")))
    model_turn_detector = _build_turn_detector()
    resolved_turn_detection = _resolve_turn_detection(tuning, model_turn_detector)

    print(
        (
            "Runtime settings: "
            f"stt_model={tuning.stt_model}, "
            f"stt_language={tuning.stt_language}, "
            f"tts_provider={tuning.tts_provider}, "
            f"tts_model={tuning.tts_model}, "
            f"tts_voice={tuning.tts_voice}, "
            f"cartesia_tts_speed={tuning.cartesia_tts_speed}, "
            f"tts_fallback_enabled={tuning.tts_fallback_enabled}, "
            f"preemptive_generation={tuning.preemptive_generation}, "
            f"turn_detection_mode={tuning.turn_detection_mode}, "
            f"resolved_turn_detection={resolved_turn_detection}"
        ),
        flush=True,
    )

    if tuning.tts_provider == "cartesia":
        cartesia_tts_engine = cartesia.TTS(
            model=tuning.tts_model,
            voice=tuning.tts_voice,
            speed=tuning.cartesia_tts_speed,
            language="en",
            word_timestamps=False,
        )
        if tuning.tts_fallback_enabled:
            # Cartesia primary with OpenAI fallback for provider-side outages.
            openai_tts_engine = openai.TTS(
                model=tuning.openai_tts_model,
                voice=tuning.openai_tts_voice,
            )
            tts_engine = lk_tts.FallbackAdapter(
                [cartesia_tts_engine, openai_tts_engine],
                max_retry_per_tts=tuning.tts_fallback_max_retry_per_provider,
            )
        else:
            tts_engine = cartesia_tts_engine
    else:
        tts_engine = openai.TTS(
            model=tuning.tts_model,
            voice=tuning.tts_voice,
        )

    stt_engine = assemblyai.STT(
        model=tuning.stt_model,
        language_detection=False,
        end_of_turn_confidence_threshold=tuning.stt_eot_confidence_threshold,
        min_turn_silence=tuning.stt_min_turn_silence_ms,
        max_turn_silence=tuning.stt_max_turn_silence_ms,
        buffer_size_seconds=tuning.stt_buffer_size_seconds,
    )
    llm_engine = openai.LLM(
        model=llm_model,
        temperature=tuning.llm_temperature,
        max_completion_tokens=tuning.llm_max_completion_tokens,
        parallel_tool_calls=False,
    )
    vad_engine = _get_vad_model()

    if tuning.use_turn_handling:
        # New API path (recommended by LiveKit) for lower-latency, production turn-taking.
        session = AgentSession(
            stt=stt_engine,
            llm=llm_engine,
            tts=tts_engine,
            vad=vad_engine,
            turn_handling={
                "turn_detection": resolved_turn_detection,
                "endpointing": {
                    "mode": "dynamic",
                    "min_delay": tuning.min_endpointing_delay,
                    "max_delay": tuning.max_endpointing_delay,
                },
                "interruption": {
                    "enabled": tuning.allow_interruptions,
                    "mode": tuning.interruption_mode,
                    "min_duration": tuning.min_interruption_duration,
                    "false_interruption_timeout": tuning.false_interruption_timeout,
                    "resume_false_interruption": tuning.resume_false_interruption,
                    "discard_audio_if_uninterruptible": True,
                },
            },
            preemptive_generation=tuning.preemptive_generation,
            min_consecutive_speech_delay=tuning.min_consecutive_speech_delay,
        )
    else:
        # Legacy path kept behind flag for rollback safety.
        session = AgentSession(
            stt=stt_engine,
            llm=llm_engine,
            tts=tts_engine,
            vad=vad_engine,
            turn_detection=resolved_turn_detection,
            preemptive_generation=tuning.preemptive_generation,
            min_endpointing_delay=tuning.min_endpointing_delay,
            max_endpointing_delay=tuning.max_endpointing_delay,
            min_consecutive_speech_delay=tuning.min_consecutive_speech_delay,
            allow_interruptions=tuning.allow_interruptions,
            min_interruption_duration=tuning.min_interruption_duration,
            false_interruption_timeout=tuning.false_interruption_timeout,
            resume_false_interruption=tuning.resume_false_interruption,
        )

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event: Any) -> None:
        item = getattr(event, "item", None)
        if item is None:
            return
        text = _extract_item_text(item)
        if not text:
            return
        transcript.append(
            TranscriptTurn(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                role=_to_role_string(getattr(item, "role", "unknown")),
                text=text,
                interrupted=bool(getattr(item, "interrupted", False)),
            )
        )

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event: Any) -> None:
        text = str(getattr(event, "transcript", "")).strip()
        if text:
            print(f"user_input_transcribed(final={bool(getattr(event, 'is_final', False))}): {text}", flush=True)

    def _finalize_once(trigger: str) -> None:
        nonlocal finalized
        if finalized:
            return
        finalized = True

        call_ended_at = datetime.now(timezone.utc)
        duration_seconds = max(0, int((call_ended_at - call_started_at).total_seconds()))
        outcome = derive_outcome(disposition)
        disposition_dict = asdict(disposition)

        print(f"Call ended. trigger={trigger} outcome={outcome}", flush=True)

        transcript_available = False
        if client_id and event_id:
            try:
                write_transcript(
                    client_id=client_id,
                    event_id=event_id,
                    transcript=transcript,
                    metadata={
                        "outcome": outcome,
                        "agent_id": agent_id,
                        "duration_seconds": duration_seconds,
                    },
                )
                transcript_available = True
            except Exception as exc:
                print(f"Transcript write failed: {exc}", flush=True)

        if client_id and event_id:
            try:
                write_event_disposition(
                    client_id=client_id,
                    event_id=event_id,
                    disposition=disposition_dict,
                    outcome=outcome,
                    duration_seconds=duration_seconds,
                    transcript_available=transcript_available,
                )
            except Exception as exc:
                print(f"Firestore write failed: {exc}", flush=True)

    @session.on("close")
    def _on_session_close(event: Any) -> None:
        reason = str(getattr(event, "reason", "unknown"))
        _finalize_once(f"session_close:{reason}")

    customer_name = str(contact.get("customer_name", "there"))
    agent_name = str(agent_config.get("agent_name", "Sophie"))
    company_name = str(agent_config.get("company_name", "our company"))

    await session.start(room=ctx.room, agent=agent)

    # Deterministic opening line gives reliable first speech.
    await session.say(
        f"Hello, this is {agent_name}. I'm a virtual representative calling from {company_name}. "
        f"May I please speak with {customer_name}?",
        add_to_chat_ctx=True,
    )

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        _finalize_once("entrypoint_finally")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="verifacto-sales-agent",
        )
    )
