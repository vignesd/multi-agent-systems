from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `streamlit run ui/app.py` from anywhere to import root modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from config import AVAILABLE_MODELS, MODEL  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from ui.chat_service import CLIENTS, AgentWorker  # noqa: E402
from ui.message_renderer import (  # noqa: E402
    collect_tool_activity,
    render_markdown,
    render_tool_activity,
)

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Multi-Agent Assistant", page_icon="🤖", layout="centered")


@st.cache_resource
def get_worker() -> AgentWorker:
    """One event loop and agent cache shared by all browser sessions."""
    return AgentWorker()


def get_chat(client: str) -> dict:
    """Per-client conversation state: raw LangChain history plus display turns."""
    chats = st.session_state.setdefault("chats", {})
    return chats.setdefault(client, {"history": [], "turns": []})


def render_turn(spec, turn: dict) -> None:
    with st.chat_message("user"):
        render_markdown(turn["query"])
    with st.chat_message("assistant", avatar=spec.icon):
        if turn.get("error"):
            st.error(turn["answer"])
            return
        render_markdown(turn["answer"])
        if turn.get("activity"):
            with st.expander("🔧 Tool activity", expanded=False):
                render_tool_activity(turn["activity"])


with st.sidebar:
    st.subheader("Settings")
    client = st.selectbox(
        "Client",
        options=list(CLIENTS),
        format_func=lambda key: f"{CLIENTS[key].icon} {CLIENTS[key].label}",
        key="client",
    )
    model_name = st.selectbox(
        "Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(MODEL),
        key="model",
    )
    spec = CLIENTS[client]
    chat = get_chat(client)

    st.caption(
        "Each client keeps its own conversation for this browser session. "
        "Changing the model keeps the history and rebuilds the agent on the next message."
    )
    if st.button("Clear this conversation", use_container_width=True):
        chat["history"].clear()
        chat["turns"].clear()
        st.rerun()

    st.divider()
    st.subheader("Example questions")
    st.markdown("\n".join(f"- {example}" for example in spec.examples))

st.title(f"{spec.icon} {spec.label}")
st.caption(spec.description)

for turn in chat["turns"]:
    render_turn(spec, turn)

query = st.chat_input(f"Ask the {spec.label}…")

if query:
    with st.chat_message("user"):
        render_markdown(query)

    turn = {"query": query}
    with st.chat_message("assistant", avatar=spec.icon):
        status = st.status(f"Working on your request ({model_name})…", expanded=True)
        try:
            answer, new_history = get_worker().run_turn(
                client, model_name, chat["history"], query
            )
            # Only the latest turn's tool calls are shown for this answer.
            activity = collect_tool_activity(new_history)
            status.update(label="Request completed", state="complete", expanded=False)
            render_markdown(answer)
            if activity:
                with st.expander("🔧 Tool activity", expanded=False):
                    render_tool_activity(activity)

            chat["history"] = new_history
            turn.update(answer=answer, activity=activity)
        except Exception as exc:
            logger.exception("%s request failed", client)
            status.update(label="Request failed", state="error", expanded=True)
            message = (
                f"The request failed: {str(exc).rstrip('.')}. Check your API keys, "
                "MCP server connections and the application logs, then try again."
            )
            st.error(message)
            turn.update(answer=message, error=True)

    chat["turns"].append(turn)
