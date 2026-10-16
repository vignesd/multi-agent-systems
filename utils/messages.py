from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage


def content_to_text(content: Any) -> str:
    """Convert common LangChain message content shapes to readable text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type in {"text", "output_text", "markdown"}:
                    value = block.get("text") or block.get("content")
                    if isinstance(value, str):
                        parts.append(value)
                elif block_type == "image":
                    parts.append("[Image content]")
                else:
                    # Keep unfamiliar blocks visible rather than silently losing them.
                    parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "\n\n".join(part for part in parts if part).strip()

    if content is None:
        return ""
    return str(content).strip()


def get_final_answer(
    result: dict[str, Any],
    default: str = "The agent did not return a text response.",
) -> str:
    """Extract the latest assistant text reply from an agent result."""
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage):
            # Skip assistant messages that only contain tool calls.
            if getattr(message, "tool_calls", None):
                continue
            answer = content_to_text(message.content)
            if answer:
                return answer

    return default
