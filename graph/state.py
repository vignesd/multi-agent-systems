from typing import Annotated, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TaskItem(TypedDict):
    """Represents an atomic task in the execution queue."""
    agent: str  # Target sub-agent: "travel", "currency", "worldclock", or "defi"
    query: str  # Isolated query for this specific task item
    completed: bool  # Tracking status flag


class AgentState(TypedDict):
    """State schema for the dynamic task-queue multi-agent workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    task_queue: List[TaskItem]  # Persistent queue of sub-tasks
    next_agent: str  # Routing key for conditional edges
    current_sub_task: Optional[str]  # Query passed strictly to active sub-agent