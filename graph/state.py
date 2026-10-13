from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TaskItem(TypedDict):
    """Represents an atomic task in the execution queue."""

    agent: str
    query: str
    status: Literal["pending", "running", "completed", "failed"]
    attempts: int
    error: Optional[str]


class AgentState(TypedDict):
    """Shared state for the supervisor-based multi-agent workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    task_queue: list[TaskItem]
    next_agent: str
    current_sub_task: Optional[str]
