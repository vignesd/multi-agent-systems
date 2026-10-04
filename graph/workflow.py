import logging
from enum import Enum
from typing import List, Optional
# from pydantic import BaseModel, Field

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agents.currency_agent import create_currency_agent
from agents.defi_agent import create_defi_agent
from agents.travel_agent import create_travel_agent
from agents.worldclock_agent import create_worldclock_agent
from config import MODEL

from graph.state import AgentState, TaskItem
from graph.models import AgentMap,SubTask,InitialPlan

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("TaskQueueWorkflow")


# ---------------------------------------------------------
# Pydantic Schemas for Task Decomposition
# ---------------------------------------------------------
# class SubTask(BaseModel):
#     agent: str = Field(
#         description="Target specialist agent. Must be one of: 'travel', 'currency', 'worldclock', 'defi'."
#     )
#     query: str = Field(
#         description="Self-contained task prompt containing ONLY the details needed for this specific step."
#     )


# class InitialPlan(BaseModel):
#     tasks: List[SubTask] = Field(
#         description="Ordered list of atomic sub-tasks decomposed from the user query."
#     )


# Agent mapping for conditional routing
# AgentMap = {
#     "travel": "travel",
#     "currency": "currency",
#     "worldclock": "worldclock",
#     "defi": "defi",
#     "finish": "final",
# }


async def build_workflow():
    model = ChatOpenAI(model=MODEL, temperature=0.0)
    
    # Model configured with structured output to guarantee valid task decomposition
    planner_model = model.with_structured_output(InitialPlan)

    # Instantiate specialist agents ONCE
    travel_agent = await create_travel_agent(model)
    currency_agent = await create_currency_agent(model)
    worldclock_agent = await create_worldclock_agent(model)
    defi_agent = await create_defi_agent(model)

    # ---------------------------------------------------------
    # Supervisor Node: Task Queue Manager & Router
    # ---------------------------------------------------------
    async def supervisor_node(state: AgentState):
        """
        State Machine Logic:
        1. If task_queue is empty, parse user request and decompose into atomic TaskItems.
        2. Scan task_queue sequentially for the first item where completed == False.
        3. If an uncompleted task exists: mark completed = True, set next_agent and current_sub_task.
        4. If all tasks are completed: set next_agent = 'finish'.
        """
        # Shallow copy queue from state or initialize empty
        task_queue: List[TaskItem] = list(state.get("task_queue") or [])

        # PHASE 1: INITIAL DECOMPOSITION (Runs only on Turn 1)
        if not task_queue and state.get("messages"):
            user_request = state["messages"][0].content
            logger.info("=" * 70)
            logger.info("[SUPERVISOR: TASK DECOMPOSITION PHASE]")
            logger.info(f"Analyzing User Request: '{user_request}'")

            planner_prompt = SystemMessage(
                content="""
You are an expert Task Planning Agent.

Decompose the user's query into an ordered list of atomic, isolated sub-tasks.
Each task MUST be assigned to one of four specialists:
1. 'travel'     : Flight searches, itineraries, location travel info.
2. 'currency'   : Exchange rates, monetary conversions.
3. 'worldclock' : Current time, time zone calculations.
4. 'defi'       : Crypto liquidity pools, risk/entry/exit signals.

CRITICAL RULES:
- If a query asks for multiple routes or items for the SAME domain (e.g., 3 different flight routes), create SEPARATE sub-tasks for each item.
- Write each 'query' as a clear, self-contained instruction. Do not rely on conversational context.
"""
            )
            
            plan: InitialPlan = await planner_model.ainvoke(
                [planner_prompt, state["messages"][0]]
            )

            # Convert Pydantic models to TaskItem dictionaries for LangGraph state
            for item in plan.tasks:
                task_queue.append(
                    {
                        "agent": item.agent.strip().lower(),
                        "query": item.query,
                        "completed": False,
                    }
                )

            logger.info(f"Generated Task Queue ({len(task_queue)} sub-tasks):")
            for idx, task in enumerate(task_queue, 1):
                logger.info(f"  Task {idx}: [{task['agent'].upper()}] -> '{task['query']}'")
            logger.info("=" * 70)

        # PHASE 2: QUEUE SELECTION & ROUTING
        next_task: Optional[TaskItem] = None
        for task in task_queue:
            if not task["completed"]:
                next_task = task
                break

        if next_task:
            # Mark task as completed so next supervisor turn moves to the next queue item
            next_task["completed"] = True
            next_agent = next_task["agent"]
            current_sub_task = next_task["query"]

            completed_count = sum(1 for t in task_queue if t["completed"])
            logger.info("[SUPERVISOR ROUTING DECISION]")
            logger.info(f"  -> Progress        : {completed_count}/{len(task_queue)} tasks executed")
            logger.info(f"  -> Routing To      : '{next_agent}'")
            logger.info(f"  -> Dispatching Query: '{current_sub_task}'")
        else:
            # All tasks in the queue have been executed
            next_agent = "finish"
            current_sub_task = None
            logger.info("[SUPERVISOR ROUTING DECISION]")
            logger.info("  -> Task Queue Exhausted. All sub-tasks completed.")
            logger.info("  -> Routing To      : 'final' (Synthesis Node)")

        return {
            "next_agent": next_agent,
            "current_sub_task": current_sub_task,
            "task_queue": task_queue,  # Updates persistent queue state
        }

    # ---------------------------------------------------------
    # Specialist Nodes (Isolated Sub-Task Execution)
    # ---------------------------------------------------------
    async def travel_node(state: AgentState):
        sub_task = state["current_sub_task"]
        logger.info(f"[NODE EXECUTION: TRAVEL] Executing: '{sub_task}'")

        # Invoke sub-agent with isolated query (prevents full history leakage)
        result = await travel_agent.ainvoke({"messages": [HumanMessage(content=sub_task)]})
        content = result["messages"][-1].content
        
        logger.info(f"[NODE EXECUTION: TRAVEL] Complete. Response length: {len(content)} chars")

        return {
            "messages": [AIMessage(content=f"Sub-Task Result ({sub_task}):\n{content}")],
        }

    async def currency_node(state: AgentState):
        sub_task = state["current_sub_task"]
        logger.info(f"[NODE EXECUTION: CURRENCY] Executing: '{sub_task}'")

        result = await currency_agent.ainvoke({"messages": [HumanMessage(content=sub_task)]})
        content = result["messages"][-1].content
        
        logger.info(f"[NODE EXECUTION: CURRENCY] Complete. Response length: {len(content)} chars")

        return {
            "messages": [AIMessage(content=f"Sub-Task Result ({sub_task}):\n{content}")],
        }

    async def worldclock_node(state: AgentState):
        sub_task = state["current_sub_task"]
        logger.info(f"[NODE EXECUTION: WORLDCLOCK] Executing: '{sub_task}'")

        result = await worldclock_agent.ainvoke({"messages": [HumanMessage(content=sub_task)]})
        content = result["messages"][-1].content
        
        logger.info(f"[NODE EXECUTION: WORLDCLOCK] Complete. Response length: {len(content)} chars")

        return {
            "messages": [AIMessage(content=f"Sub-Task Result ({sub_task}):\n{content}")],
        }

    async def defi_node(state: AgentState):
        sub_task = state["current_sub_task"]
        logger.info(f"[NODE EXECUTION: DEFI] Executing: '{sub_task}'")

        result = await defi_agent.ainvoke({"messages": [HumanMessage(content=sub_task)]})
        content = result["messages"][-1].content
        
        logger.info(f"[NODE EXECUTION: DEFI] Complete. Response length: {len(content)} chars")

        return {
            "messages": [AIMessage(content=f"Sub-Task Result ({sub_task}):\n{content}")],
        }

    # ---------------------------------------------------------
    # Final Synthesis Node
    # ---------------------------------------------------------
    async def final_node(state: AgentState):
        logger.info("[FINAL SYNTHESIS NODE] Compiling final unified output for user...")

        final_prompt = SystemMessage(
            content="""
You are the final response agent.

Synthesize all collected specialist sub-task outputs from the conversation history into a clear, unified, and well-structured answer.

Rules:
- Directly answer all parts of the original user prompt.
- Organize complex or multi-item outputs using clean headings or bullet points.
- Never mention internal agent names, sub-task structures, or orchestration mechanics.
- Do not use phrases like "Based on sub-task results". Present facts naturally.
"""
        )
        response = await model.ainvoke([final_prompt, *state["messages"]])
        logger.info("[FINAL SYNTHESIS NODE] Final response successfully generated.")
        return {"messages": [response]}

    # ---------------------------------------------------------
    # Graph Construction
    # ---------------------------------------------------------
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("travel", travel_node)
    workflow.add_node("currency", currency_node)
    workflow.add_node("worldclock", worldclock_node)
    workflow.add_node("defi", defi_node)
    workflow.add_node("final", final_node)

    # Set Entry Point
    workflow.add_edge(START, "supervisor")

    # Conditional Routing based on supervisor's next_agent
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        AgentMap,
    )

    # Specialist nodes always loop back to supervisor to process remaining queue items
    workflow.add_edge("travel", "supervisor")
    workflow.add_edge("currency", "supervisor")
    workflow.add_edge("worldclock", "supervisor")
    workflow.add_edge("defi", "supervisor")

    # Final Node terminates execution
    workflow.add_edge("final", END)

    return workflow.compile()