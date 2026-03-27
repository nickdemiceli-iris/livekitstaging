from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from prompts.collections import build_system_prompt


@dataclass
class CollectionsDispositionState:
    customer_reached: bool = False
    customer_verified: bool = False
    payment_intent: str | None = None
    promised_payment_amount_usd: float | None = None
    promised_payment_date: str | None = None
    hardship_flag: bool = False
    hardship_reason: str | None = None
    next_step: str | None = None
    next_step_datetime: str | None = None
    best_phone_number: str | None = None
    notes: str = ""


def _as_money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:.2f}"
    return "N/A"


def _build_customer_context(contact: dict[str, Any]) -> str:
    lines = [
        f"- Customer Name: {contact.get('customer_name', 'Unknown')}",
        f"- Phone Number: {contact.get('phone_number', 'N/A')}",
        f"- State: {contact.get('state', 'Unknown')}",
        f"- Due Amount: {_as_money(contact.get('due_amount_usd'))}",
        f"- Due Date: {contact.get('due_date', 'N/A')}",
        f"- Account Status: {contact.get('account_status', 'N/A')}",
        f"- Today's Date: {date.today().isoformat()}",
    ]
    return "\n".join(lines)


def derive_outcome(disposition: CollectionsDispositionState) -> str:
    if not disposition.customer_reached:
        return "not_reached"
    if disposition.hardship_flag:
        return "hardship_review"
    intent = (disposition.payment_intent or "").strip().lower()
    if intent in {"paid_now", "pay_now"}:
        return "payment_now"
    if intent in {"promise_to_pay", "promise", "scheduled_payment"}:
        return "promise_to_pay"
    if intent in {"refused", "disputed"}:
        return intent
    if disposition.next_step:
        return "collections_follow_up"
    return "collections_incomplete"


def build_collections_agent(
    contact: dict[str, Any],
    agent_config: dict[str, Any],
) -> tuple["CollectionsAgent", CollectionsDispositionState]:
    agent_name = str(agent_config.get("agent_name", "Ava"))
    company_name = str(agent_config.get("company_name", "Simple Loans"))
    customer_name = str(contact.get("customer_name", "there"))
    due_amount_usd = str(contact.get("due_amount_usd", ""))
    due_date = str(contact.get("due_date", ""))
    opening_line = str(agent_config.get("opening_line", "")).strip()
    customer_context = _build_customer_context(contact)

    system_prompt = str(agent_config.get("system_prompt", "")).strip()
    if system_prompt:
        try:
            prompt = system_prompt.format(
                agent_name=agent_name,
                company_name=company_name,
                customer_name=customer_name,
                due_amount_usd=due_amount_usd,
                due_date=due_date,
                opening_line=opening_line,
                customer_context=customer_context,
            )
        except KeyError as exc:
            print(f"Missing prompt variable {exc} - using raw template", flush=True)
            prompt = system_prompt
    else:
        prompt = build_system_prompt(
            customer_context=customer_context,
            customer_name=customer_name,
            agent_name=agent_name,
            company_name=company_name,
            due_amount_usd=due_amount_usd,
            due_date=due_date,
            opening_line=opening_line,
        )

    disposition = CollectionsDispositionState()
    return CollectionsAgent(instructions=prompt, disposition=disposition), disposition


class CollectionsAgent(Agent):
    def __init__(self, instructions: str, disposition: CollectionsDispositionState) -> None:
        super().__init__(instructions=instructions)
        self._disposition = disposition

    @function_tool
    async def mark_collections_contact(
        self,
        context: RunContext,
        reached_customer: bool,
        customer_verified: bool,
        note: str = "",
    ) -> str:
        self._disposition.customer_reached = reached_customer
        self._disposition.customer_verified = customer_verified
        if note:
            self._append_note(note)
        return "Collections contact status recorded."

    @function_tool
    async def mark_payment_intent(
        self,
        context: RunContext,
        payment_intent: str,
        promised_payment_amount_usd: float = 0.0,
        promised_payment_date: str = "",
        best_phone_number: str = "",
        note: str = "",
    ) -> str:
        self._disposition.payment_intent = payment_intent.strip().lower()
        if promised_payment_amount_usd > 0:
            self._disposition.promised_payment_amount_usd = float(promised_payment_amount_usd)
        if promised_payment_date.strip():
            self._disposition.promised_payment_date = promised_payment_date.strip()
        if best_phone_number.strip():
            self._disposition.best_phone_number = best_phone_number.strip()
        if note:
            self._append_note(note)
        return "Payment intent recorded."

    @function_tool
    async def mark_hardship(
        self,
        context: RunContext,
        hardship_flag: bool,
        hardship_reason: str = "",
    ) -> str:
        self._disposition.hardship_flag = hardship_flag
        if hardship_reason:
            self._disposition.hardship_reason = hardship_reason.strip()
            self._append_note(hardship_reason)
        return "Hardship details recorded."

    @function_tool
    async def mark_next_step(
        self,
        context: RunContext,
        next_step: str,
        next_step_datetime: str = "",
        best_phone_number: str = "",
        note: str = "",
    ) -> str:
        self._disposition.next_step = next_step.strip().lower()
        if next_step_datetime.strip():
            self._disposition.next_step_datetime = next_step_datetime.strip()
        if best_phone_number.strip():
            self._disposition.best_phone_number = best_phone_number.strip()
        if note:
            self._append_note(note)
        return "Collections next-step details recorded."

    def _append_note(self, note: str) -> None:
        clean = note.strip()
        if not clean:
            return
        if self._disposition.notes:
            self._disposition.notes += f" | {clean}"
        else:
            self._disposition.notes = clean
