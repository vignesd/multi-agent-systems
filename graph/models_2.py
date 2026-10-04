from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class AgentRoute(str, Enum):
    TRAVEL = "travel"
    CURRENCY = "currency"
    WORLDCLOCK = "worldclock"
    DEFI = "defi"
    FINISH = "finish"


class SubTask(BaseModel):
    agent: str = Field(
        description="Target specialist agent. Must be one of: 'travel', 'currency', 'worldclock', 'defi'."
    )
    query: str = Field(
        description="Self-contained task prompt containing ONLY the details needed for this specific step."
    )


class InitialPlan(BaseModel):
    is_supported: bool = Field(
        description="Set to True ONLY if the user request falls under travel, currency, worldclock, or defi. Set to False for all other topics (e.g. general tech support, phone reset, cooking, general coding)."
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Explanation if is_supported is False (e.g., 'Request is out of scope for available agents.')"
    )
    tasks: List[SubTask] = Field(
        default_factory=list,
        description="List of valid sub-tasks. MUST be empty if is_supported is False. NEVER create 'N/A' or dummy tasks."
    )

    AgentMap = {
    "travel": "travel",
    "currency": "currency",
    "worldclock": "worldclock",
    "defi": "defi",
    "finish": "final",
}