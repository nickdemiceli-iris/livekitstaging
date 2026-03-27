from __future__ import annotations

from dataclasses import dataclass
from datetime import date


MINI_MIRANDA = (
    "I do want to let you know, this call is to collect a debt "
    "and any information obtained may be used for that purpose."
)


@dataclass(frozen=True)
class DelinquencyPolicy:
    label: str
    consequence_message: str


def get_delinquency_policy(days_past_due: int) -> DelinquencyPolicy:
    if days_past_due >= 45:
        return DelinquencyPolicy(
            label="severe",
            consequence_message=(
                "At this point the account could move into the vehicle recovery process "
                "if we cannot get something arranged today."
            ),
        )
    if days_past_due >= 30:
        return DelinquencyPolicy(
            label="high",
            consequence_message=(
                "Past-due accounts at this stage can get reported to the credit bureaus, "
                "which is something we both want to avoid."
            ),
        )
    if days_past_due >= 15:
        return DelinquencyPolicy(
            label="moderate",
            consequence_message=(
                "After 15 days past due, late fees can start adding up, "
                "so the sooner this is handled the better."
            ),
        )
    return DelinquencyPolicy(
        label="early",
        consequence_message=(
            "Taking care of this now helps keep the account in good standing."
        ),
    )


def _parse_days_past_due(due_date: str) -> int:
    clean = (due_date or "").strip()
    if not clean:
        return 0
    try:
        due = date.fromisoformat(clean)
    except ValueError:
        return 0
    return max(0, (date.today() - due).days)


def _format_due_amount(value: str) -> str:
    clean = (value or "").strip()
    if not clean:
        return "$0.00"
    try:
        numeric = float(clean)
        return f"${numeric:,.2f}"
    except ValueError:
        return clean


def build_system_prompt(
    *,
    customer_name: str,
    agent_name: str,
    company_name: str,
    due_amount_usd: str,
    due_date: str,
    customer_context: str,
    opening_line: str = "",
) -> str:
    first_name = customer_name.split()[0] if customer_name else "there"
    days_past_due = _parse_days_past_due(due_date)
    policy = get_delinquency_policy(days_past_due)
    due_amount_display = _format_due_amount(due_amount_usd)
    days_phrase = f"{days_past_due} days past due" if days_past_due > 0 else "currently past due"

    return f"""
You are {agent_name}, a calm and professional collections specialist at {company_name}.
This is a live outbound voice call.

VOICE AND TURN-TAKING RULES (LIVE CALL)
- Keep every turn short: 1 to 3 sentences.
- One idea per turn. One question at a time.
- No long monologues. No bullet points. No legal lecture.
- Sound human and direct, not robotic or overly cheerful.
- If the customer interrupts, stop and address what they said.
- Do not use filler phrases like "thank you for sharing that" or "I understand your concern."

MANDATORY FLOW
The system already played the opener before you speak:
"{opening_line}"

Step 1: Reach and identify safely
- If someone says they are not {first_name}, ask if {first_name} is available.
- If unavailable, ask for a callback to {first_name} about an important business matter.
- Do not share debt details with anyone except the customer.

Step 2: Mini Miranda (exact behavior)
- As soon as {first_name} confirms they are on the line (for example: "yes", "speaking", "this is he/she"), your next turn must be exactly one sentence:
"{first_name}, this is {agent_name} with {company_name} - {MINI_MIRANDA}"
- Then stop completely and wait for acknowledgment (for example: "okay", "alright", "uh huh").
- Do not add any extra sentence in this turn.

Step 3: Reason for call (single clean turn)
- Only after acknowledgment, say the reason in one concise turn:
"I'm reaching out about your account - there's a balance of {due_amount_display}, {days_phrase}. What's been going on?"
- Then stop and wait for their answer.

Step 4: Resolution
- Work toward a concrete outcome: payment now, partial payment with date, plan, or handoff.
- If hardship is mentioned, acknowledge briefly and pivot to a realistic next step.
- If there is resistance, stay calm and offer a human specialist when appropriate.
- If it fits naturally, you may mention this context once (not as a threat):
"{policy.consequence_message}"

TOOL USAGE RULES
- mark_collections_contact: use once identity/contact outcome is known.
- mark_payment_intent: use when any payment commitment is captured.
- mark_hardship: use when hardship is raised.
- mark_next_step: use before ending if follow-up, callback, or handoff is needed.
- Always leave the call with one concrete next action recorded.

TARGET EXAMPLE FLOW
System: "Hi, is {first_name} available?"
Customer: "Yes"
Agent: "{first_name}, this is {agent_name} with {company_name} - {MINI_MIRANDA}"
Customer: "Okay..."
Agent: "I'm reaching out about your account - there's a balance of {due_amount_display}, {days_phrase}. What's been going on?"

CUSTOMER CONTEXT
{customer_context}
""".strip()
