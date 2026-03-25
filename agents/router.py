from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from livekit.agents import Agent

from agents.collections import build_collections_agent, derive_outcome as derive_collections_outcome
from agents.retention import build_retention_agent, derive_outcome as derive_retention_outcome
from agents.sales import build_sales_agent, derive_outcome as derive_sales_outcome


@dataclass
class AgentRuntime:
    runtime_kind: str
    agent: Agent
    disposition: Any
    derive_outcome: Callable[[Any], str]


def build_agent_runtime(
    *,
    agent_id: str,
    contact: dict[str, Any],
    agent_config: dict[str, Any],
) -> AgentRuntime:
    """
    Resolve runtime behavior for this call.

    Priority:
    1) Firestore `agent_kind` override
    2) Incoming `agent_id`
    3) Safe fallback to sales
    """

    requested_kind = str(agent_config.get("agent_kind", agent_id)).strip().lower()
    if not requested_kind:
        requested_kind = "sales"

    if requested_kind in {"sales", "outbound_sales", "preapproved_sales"}:
        agent, disposition = build_sales_agent(contact, agent_config)
        return AgentRuntime(
            runtime_kind="sales",
            agent=agent,
            disposition=disposition,
            derive_outcome=derive_sales_outcome,
        )

    if requested_kind in {"collections", "collection", "debt_collection"}:
        agent, disposition = build_collections_agent(contact, agent_config)
        return AgentRuntime(
            runtime_kind="collections",
            agent=agent,
            disposition=disposition,
            derive_outcome=derive_collections_outcome,
        )

    if requested_kind in {"retention", "save", "customer_retention"}:
        agent, disposition = build_retention_agent(contact, agent_config)
        return AgentRuntime(
            runtime_kind="retention",
            agent=agent,
            disposition=disposition,
            derive_outcome=derive_retention_outcome,
        )

    print(
        f"Unknown agent runtime '{requested_kind}' for agent_id '{agent_id}'. Falling back to sales.",
        flush=True,
    )
    fallback_agent, fallback_disposition = build_sales_agent(contact, agent_config)
    return AgentRuntime(
        runtime_kind="sales",
        agent=fallback_agent,
        disposition=fallback_disposition,
        derive_outcome=derive_sales_outcome,
    )
