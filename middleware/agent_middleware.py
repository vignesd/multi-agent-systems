
from langchain.agents.middleware import (
    SummarizationMiddleware,
    PIIMiddleware,
    AgentMiddleware,
    before_agent,
)
from langchain.messages import AIMessage

from prompts.system_prompts import (
    TRAVEL_SUMMARY_PROMPT,
    CHINOOK_SUMMARY_PROMPT,
    GENERIC_SUMMARY_PROMPT
)

import re


def get_summary_middleware(MODEL, Middleware_Prompt: str = GENERIC_SUMMARY_PROMPT):

    return SummarizationMiddleware(
        model=MODEL,
        trigger=("messages", 50),
        keep=("messages", 10),
        summary_prompt=Middleware_Prompt,
        trim_tokens_to_summarize=5000,
    )


def get_phone_number_middleware():
    return PIIMiddleware(
        "phone_number",
        # detector=r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}",
        detector=(
            r"\+?\d{1,3}\s*"
            r"(?:\(\s*\d{2,4}\s*\)|\d{2,4})"
            r"(?:[\s.-]*\d{2,4}){1,3}"
        ),
        strategy="mask",
        apply_to_output=True,
        apply_to_input=False,
        apply_to_tool_results=True
    )


def get_email_middleware():

    return PIIMiddleware(
        "email",
        strategy="mask",          # Options: 'mask', 'redact', 'block', 'hash'
        apply_to_input=False,
        apply_to_output=True
    )


INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"reveal (your )?(system prompt|hidden instructions)",
    r"show (me )?(your )?(system prompt|hidden instructions)",
    r"act as (an? )?(unrestricted|unfiltered) assistant",
    r"bypass (your )?(safety|security|restrictions)",
    r"ignore your (rules|guidelines|instructions)",
    r"reveal (api keys|passwords|secrets|credentials)",
]

COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in INJECTION_PATTERNS
]


def contains_prompt_injection(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in COMPILED_PATTERNS
    )


@before_agent
def get_prompt_injection_guard(state, runtime):
    """Reject obvious prompt-injection attempts before agent execution."""

    messages = state.get("messages", [])

    if not messages:
        return None

    last_message = messages[-1]

    if getattr(last_message, "type", None) != "human":
        return None

    content = last_message.content

    if not isinstance(content, str):
        return None

    if contains_prompt_injection(content):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I can't follow instructions that attempt to "
                        "override my operating rules or reveal "
                        "confidential information. Please ask a question "
                        "within my supported capabilities."
                    )
                )
            ]
        }

    return None
