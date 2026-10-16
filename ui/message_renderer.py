from __future__ import annotations

import json
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from utils.messages import content_to_text


def render_markdown(content: Any) -> None:
    """Render assistant/user content as safe Streamlit Markdown."""
    text = content_to_text(content)
    if text:
        # unsafe_allow_html defaults to False; do not enable it for model output.
        st.markdown(text)
    else:
        st.caption("(No text content in this message.)")


def collect_tool_activity(messages: list[Any]) -> list[dict[str, str]]:
    """Summarize tool calls and results from the latest turn only."""
    # Histories accumulate across turns; only show what happened after the last user message.
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            start = index + 1
            break

    activity: list[dict[str, str]] = []

    for message in messages[start:]:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = str(call.get("name", "Tool"))
            args = call.get("args", {})
            activity.append({
                "kind": "call",
                "name": name,
                "detail": json.dumps(args, ensure_ascii=False, indent=2, default=str),
                "status": "Requested",
            })

        # ToolMessage has a tool_call_id and usually a name.
        if getattr(message, "tool_call_id", None):
            name = str(getattr(message, "name", None) or "Tool result")
            activity.append({
                "kind": "result",
                "name": name,
                "detail": content_to_text(getattr(message, "content", "")),
                "status": "Completed",
            })

    return activity


def render_tool_activity(activity: list[dict[str, str]]) -> None:
    if not activity:
        st.caption("No tool calls were exposed in the returned message history.")
        return

    for index, item in enumerate(activity, start=1):
        title = f"{index}. {item['name']} — {item['status']}"
        with st.expander(title, expanded=False):
            st.markdown(f"**Event:** {item['kind'].title()}")
            st.code(item["detail"] or "(No details returned)", language="json" if item["kind"] == "call" else "text")
