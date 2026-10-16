"""Agent middleware used by every specialist agent.

Middleware is organised in two layers:

1. ``get_default_middleware(model)`` - a shared reliability stack attached to
   every agent: call limits, model retry/fallback, context trimming, tool
   auditing, tool retry and tool error handling.
2. Agent-specific extras - conversation summarization, PII masking and the
   prompt-injection guard - added only by the agents that need them.

Ordering matters: in ``create_agent(middleware=[...])`` the first entry is the
outermost wrapper and the last entry is the innermost (closest to the actual
model or tool call). ``before_*`` hooks run in list order.
"""

import logging
import re
import time

from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    before_agent,
    wrap_tool_call,
)
from langchain.messages import AIMessage

from config import FALLBACK_MODEL
from prompts.system_prompts import (
    TRAVEL_SUMMARY_PROMPT,
    CHINOOK_SUMMARY_PROMPT,
    GENERIC_SUMMARY_PROMPT
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Shared reliability stack
# ---------------------------------------------------------

@wrap_tool_call
async def tool_audit_logger(request, handler):
    """Log every tool call's name, duration and outcome.

    Arguments and results are deliberately not logged: tool payloads can
    contain customer PII, and LangSmith tracing already captures them when
    enabled.
    """
    name = request.tool_call.get("name", "unknown")
    started = time.perf_counter()
    try:
        result = await handler(request)
    except Exception as exc:
        logger.warning(
            "Tool %s raised %s after %.2fs",
            name, type(exc).__name__, time.perf_counter() - started,
        )
        raise

    status = getattr(result, "status", "success")
    logger.info(
        "Tool %s finished (%s) in %.2fs",
        name, status, time.perf_counter() - started,
    )
    return result


def _tool_error_message(exc: Exception, request) -> str:
    # Name only the exception type: raw messages may expose URLs, keys or
    # internal details to the model and, through it, to the user.
    return (
        f"Tool failed with {type(exc).__name__}. The service may be "
        "temporarily unavailable. Tell the user this part could not be "
        "retrieved; do not invent a result."
    )


def get_default_middleware(model) -> list[AgentMiddleware]:
    """Return the reliability stack every agent should run with.

    Listed outermost -> innermost.
    """
    middleware: list[AgentMiddleware] = [
        ModelCallLimitMiddleware(run_limit=10, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=15),
    ]  # type:ignore

    # Switches to a backup model if the primary keeps failing (outage, rate
    # limit). Sits outside the retry so each model gets its own retries first.
    primary = getattr(model, "model_name", None)
    if FALLBACK_MODEL and FALLBACK_MODEL != primary:
        middleware.append(ModelFallbackMiddleware(f"openai:{FALLBACK_MODEL}"))

    middleware += [
        # Retries transient model errors (timeouts, 5xx, rate limits) with
        # exponential backoff and jitter.
        ModelRetryMiddleware(max_retries=2),

        # Replaces old tool results (invoice lists, crawled pages) with a
        # placeholder once the context grows past ~20k tokens, keeping the
        # latest 3 intact. Cuts tokens long before summarization kicks in.
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=20_000, keep=3)],
        ),

        # Logs each tool call's final outcome, after retries and error handling.
        tool_audit_logger,

        # Converts a tool exception into an error ToolMessage so the model can
        # recover or report the failure instead of the whole run crashing.
        ToolErrorMiddleware(on_error=_tool_error_message),

        # Innermost: retries transient MCP failures first. on_failure="error"
        # re-raises the last exception so ToolErrorMiddleware above handles it.
        ToolRetryMiddleware(max_retries=2, on_failure="error"),
    ]
    return middleware


# ---------------------------------------------------------
# Conversation summarization
# ---------------------------------------------------------

def get_summary_middleware(MODEL, Middleware_Prompt: str = GENERIC_SUMMARY_PROMPT):
    """Summarize long conversations so follow-ups keep context without
    overflowing the model window.

    Triggers at 50 messages, keeps the latest 10 verbatim, and summarizes the
    rest with a domain-specific prompt.
    """
    return SummarizationMiddleware(
        model=MODEL,
        trigger=("messages", 50),
        keep=("messages", 10),
        summary_prompt=Middleware_Prompt,
        trim_tokens_to_summarize=5000,
    )


# ---------------------------------------------------------
# PII masking
# ---------------------------------------------------------

PHONE_NUMBER_PATTERN = (
    r"\+\d{1,3}[\s.-]*(?:\(\d{1,4}\)[\s.-]*)?\d{2,5}(?:[\s.-]*\d{2,5}){1,3}"
    r"|\(\d{2,4}\)\s*\d{3,4}[\s.-]?\d{4}"
    r"|\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"
)


def get_phone_number_middleware():
    """Mask phone numbers in model output and tool results."""
    return PIIMiddleware(
        "phone_number",
        detector=PHONE_NUMBER_PATTERN,
        strategy="mask",
        apply_to_output=True,
        apply_to_input=False,
        apply_to_tool_results=True
    )


def get_email_middleware():
    """Mask email addresses in model output and tool results.

    Tool results are masked too, so raw customer emails returned by the MCP
    server never reach the model.
    """
    return PIIMiddleware(
        "email",
        strategy="mask",          # Options: 'mask', 'redact', 'block', 'hash'
        apply_to_input=False,
        apply_to_output=True,
        apply_to_tool_results=True
    )


# ---------------------------------------------------------
# Prompt-injection guard
# ---------------------------------------------------------

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


@before_agent(can_jump_to=["end"])
def get_prompt_injection_guard(state, runtime):
    """Reject obvious prompt-injection attempts before the agent runs.

    On a match, a refusal is returned and the run jumps straight to the end,
    so the model never sees the injected instruction. This is a pattern-based
    first line of defence; scope is also enforced in each system prompt.
    """

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
            ],
            "jump_to": "end",
        }

    return None
