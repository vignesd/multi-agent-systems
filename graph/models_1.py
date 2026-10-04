from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Pydantic Schemas for Task Decomposition
# ---------------------------------------------------------

class SubTask(BaseModel):
    agent: str = Field(
        description="Target specialist agent. Must be one of: 'travel', 'currency', 'worldclock', 'defi'."
    )
    query: str = Field(
        description="Self-contained task prompt containing ONLY the details needed for this specific step."
    )


class InitialPlan(BaseModel):
    tasks: List[SubTask] = Field(
        description="Ordered list of atomic sub-tasks decomposed from the user query."
    )

AgentMap = {
    "travel": "travel",
    "currency": "currency",
    "worldclock": "worldclock",
    "defi": "defi",
    "finish": "final",
}



