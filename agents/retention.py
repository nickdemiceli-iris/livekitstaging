from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from prompts.retention import build_system_prompt


@dataclass
class RetentionDispositionState:
    customer_reached: bool = False
    at_risk_reason: str | None = None
    accepted_offer: bool | None = None
    requested_callback: bool = False
    next_step: str | None = None
    next_step_datetime: str | None = None
    best_phone_number: str | None = None
    notes: str = ""


def derive_outcome(disposition: RetentionDispositionState) -> str:
    if not disposition.customer_reached:
        return "not_reached"
    if disposition.accepted_offer is True:
        return "retained"
    if disposition.requested_callback:
        return "callback_scheduled"
    if disposition.accepted_offer is False:
        return "churn_risk_open"
    return "retention_incomplete"


def build_retention_agent(
    contact: dict[str, Any],
    agent_config: dict[str, Any],
) -> tuple["RetentionAgent", RetentionDispositionState]:
    agent_name = str(agent_config.get("agent_name", "Ava"))
    company_name = str(agent_config.get("company_name", "Simple Loans"))
    customer_name = str(contact.get("customer_name", "there"))
    product_name = str(contact.get("product_name", "your current plan"))

    customer_context_lines = [
        f"- Customer Name: {customer_name}",
        f"- Product: {product_name}",
        f"- State: {contact.get('state', 'Unknown')}",
        f"- Phone Number: {contact.get('phone_number', 'N/A')}",
        f"- Last Interaction Summary: {contact.get('last_interaction_summary', 'N/A')}",
        f"- Churn Risk Signal: {contact.get('churn_risk_signal', 'N/A')}",
    ]
    customer_context = "\n".join(customer_context_lines)

    system_prompt = str(agent_config.get("system_prompt", "")).strip()
    if system_prompt:
        try:
            prompt = system_prompt.format(
                agent_name=agent_name,
                company_name=company_name,
                customer_name=customer_name,
                product_name=product_name,
                customer_context=customer_context,
            )
        except KeyError as exc:
            print(f"Missing prompt variable {exc} - using raw template", flush=True)
            prompt = system_prompt
    else:
        prompt = build_system_prompt(
            agent_name=agent_name,
            company_name=company_name,
            customer_name=customer_name,
            product_name=product_name,
            customer_context=customer_context,
        )

    disposition = RetentionDispositionState()
    return RetentionAgent(instructions=prompt, disposition=disposition), disposition


class RetentionAgent(Agent):
    def __init__(self, instructions: str, disposition: RetentionDispositionState) -> None:
        super().__init__(instructions=instructions)
        self._disposition = disposition

    @function_tool
    async def mark_retention_status(
        self,
        context: RunContext,
        reached_customer: bool,
        accepted_offer: bool | None = None,
        at_risk_reason: str = "",
        note: str = "",
    ) -> str:
        self._disposition.customer_reached = reached_customer
        self._disposition.accepted_offer = accepted_offer
        if at_risk_reason:
            self._disposition.at_risk_reason = at_risk_reason.strip()
        if note:
            self._append_note(note)
        return "Retention status recorded."

    @function_tool
    async def mark_next_step(
        self,
        context: RunContext,
        requested_callback: bool,
        next_step: str,
        next_step_datetime: str,
        best_phone_number: str,
        note: str = "",
    ) -> str:
        self._disposition.requested_callback = requested_callback
        self._disposition.next_step = next_step.strip().lower()
        self._disposition.next_step_datetime = next_step_datetime.strip()
        self._disposition.best_phone_number = best_phone_number.strip()
        if note:
            self._append_note(note)
        return "Next-step details recorded."

    def _append_note(self, note: str) -> None:
        clean = note.strip()
        if not clean:
            return
        if self._disposition.notes:
            self._disposition.notes += f" | {clean}"
        else:
            self._disposition.notes = clean
