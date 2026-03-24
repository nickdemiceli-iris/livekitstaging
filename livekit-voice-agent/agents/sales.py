from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from prompts.sales import build_system_prompt


@dataclass
class CallDispositionState:
    customer_reached: bool = False
    customer_interested: bool | None = None
    not_interested_reason: str | None = None
    requested_loan_amount_usd: float | None = None
    referred_to_loan_officer: bool = False
    qualification_notes: str | None = None
    next_step: str | None = None
    next_step_datetime: str | None = None
    best_phone_number: str | None = None
    notes: str = ""


def _as_money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:.2f}"
    return "N/A"


def build_customer_context(contact: dict[str, Any]) -> str:
    campaign_type = str(contact.get("campaign_type", "pre_approved")).lower()
    script_version = str(contact.get("script_version", "A")).upper()

    lines = [
        f"- Customer Name: {contact.get('customer_name', 'Unknown')}",
        (
            "- Vehicle: "
            f"{contact.get('vehicle_year', 'N/A')} "
            f"{contact.get('vehicle_make', 'N/A')} "
            f"{contact.get('vehicle_model', 'N/A')}"
        ),
        f"- Campaign Type: {campaign_type}",
        f"- Script Version: {script_version if campaign_type == 'pre_approved' else 'N/A'}",
        f"- State: {contact.get('state', 'Unknown')}",
        f"- Lead Source: {contact.get('lead_source', 'N/A')}",
        f"- Phone Number: {contact.get('phone_number', 'N/A')}",
        f"- Contact Preference: {contact.get('contact_channel_preference', 'phone')}",
        f"- Language Preference: {contact.get('language_preference', 'English')}",
        f"- Today's Date: {date.today().isoformat()}",
    ]
    if campaign_type == "pre_approved":
        lines.extend(
            [
                f"- Pre-Approved Amount: {_as_money(contact.get('advance_amount_usd'))}",
                f"- Max Term Months: {contact.get('max_term_months', 12)}",
                f"- VDC Total (12 months): {_as_money(contact.get('vdc_total_usd'))}",
                (
                    "- Estimated Monthly Payment (principal + VDC + interest): "
                    f"{_as_money(contact.get('estimated_monthly_payment_usd'))}"
                ),
                (
                    "- Estimated Total Interest (12 months): "
                    f"{_as_money(contact.get('estimated_total_interest_usd'))}"
                ),
            ]
        )
    return "\n".join(lines)


def derive_outcome(disposition: CallDispositionState) -> str:
    if not disposition.customer_reached:
        return "not_reached"
    if disposition.customer_interested is False:
        return "not_interested"
    if disposition.referred_to_loan_officer:
        return "loan_officer_referral"
    if disposition.next_step:
        if "inspection" in disposition.next_step:
            return "inspection_scheduled"
        if "callback" in disposition.next_step:
            return "callback_scheduled"
    if disposition.customer_interested is True:
        return "interested_pending_next_step"
    return "incomplete"


def build_sales_agent(
    contact: dict[str, Any],
    agent_config: dict[str, Any],
) -> tuple["SalesAgent", CallDispositionState]:
    agent_name = str(agent_config.get("agent_name", "Abby"))
    company_name = str(agent_config.get("company_name", "Simple Loans"))
    campaign_type = str(contact.get("campaign_type", agent_config.get("campaign_type", "pre_approved"))).lower()
    script_version = str(contact.get("script_version", agent_config.get("script_version", "A"))).upper()
    if campaign_type != "pre_approved":
        script_version = "N/A"

    customer_name = str(contact.get("customer_name", "there"))
    vehicle_year = str(contact.get("vehicle_year", ""))
    vehicle_make = str(contact.get("vehicle_make", ""))
    vehicle_model = str(contact.get("vehicle_model", ""))
    advance_amount_usd = str(contact.get("advance_amount_usd", ""))

    customer_context = build_customer_context(contact)

    system_prompt = str(agent_config.get("system_prompt", "")).strip()
    if system_prompt:
        try:
            prompt = system_prompt.format(
                agent_name=agent_name,
                company_name=company_name,
                customer_name=customer_name,
                vehicle_year=vehicle_year,
                vehicle_make=vehicle_make,
                vehicle_model=vehicle_model,
                advance_amount_usd=advance_amount_usd,
                campaign_type=campaign_type,
                customer_context=customer_context,
            )
        except KeyError as e:
            print(f"Missing prompt variable {e} - using raw template", flush=True)
            prompt = system_prompt
    else:
        prompt = build_system_prompt(
            customer_context=customer_context,
            campaign_type=campaign_type,
            script_version=script_version,
            customer_name=customer_name,
            agent_name=agent_name,
            company_name=company_name,
            vehicle_year=vehicle_year,
            vehicle_make=vehicle_make,
            vehicle_model=vehicle_model,
            advance_amount_usd=advance_amount_usd,
        )

    disposition = CallDispositionState()
    return SalesAgent(instructions=prompt, disposition=disposition), disposition


class SalesAgent(Agent):
    def __init__(self, instructions: str, disposition: CallDispositionState) -> None:
        super().__init__(instructions=instructions)
        self._disposition = disposition

    @function_tool
    async def mark_interest_outcome(
        self,
        context: RunContext,
        reached_customer: bool,
        interested: bool | None = None,
        reason: str = "",
        note: str = "",
    ) -> str:
        self._disposition.customer_reached = reached_customer
        self._disposition.customer_interested = interested
        if interested is False and reason:
            self._disposition.not_interested_reason = reason.strip()
        if note:
            self._append_note(note)
        return "Interest outcome recorded."

    @function_tool
    async def mark_requested_loan_amount(
        self,
        context: RunContext,
        requested_loan_amount_usd: float,
        note: str = "",
    ) -> str:
        if requested_loan_amount_usd > 0:
            self._disposition.requested_loan_amount_usd = float(requested_loan_amount_usd)
        if note:
            self._append_note(note)
        return "Requested loan amount recorded."

    @function_tool
    async def mark_loan_officer_referral(
        self,
        context: RunContext,
        reason: str = "",
    ) -> str:
        self._disposition.referred_to_loan_officer = True
        if reason:
            self._append_note(reason)
        return "Loan officer referral recorded."

    @function_tool
    async def mark_qualification_notes(
        self,
        context: RunContext,
        qualification_notes: str,
    ) -> str:
        self._disposition.qualification_notes = qualification_notes.strip()
        return "Qualification notes recorded."

    @function_tool
    async def mark_next_step(
        self,
        context: RunContext,
        next_step: str,
        next_step_datetime: str,
        best_phone_number: str,
        note: str = "",
    ) -> str:
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
