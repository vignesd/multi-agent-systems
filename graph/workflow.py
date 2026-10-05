import logging
from typing import List, Optional

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agents.currency_agent import create_currency_agent
from agents.defi_agent import create_defi_agent
from agents.travel_agent import create_travel_agent
from agents.worldclock_agent import create_worldclock_agent
from agents.websearch_agent import create_websearch_agent
from config import MODEL

from graph.models import AgentMap, InitialPlan, SubTask
from graph.state import AgentState, TaskItem

logger = logging.getLogger(__name__)


async def build_workflow():
    model = ChatOpenAI(model=MODEL, temperature=0.0)

    # Model configured with structured output to guarantee valid task decomposition
    planner_model = model.with_structured_output(InitialPlan)

    # Instantiate specialist agents ONCE
    travel_agent = await create_travel_agent(model)
    currency_agent = await create_currency_agent(model)
    worldclock_agent = await create_worldclock_agent(model)
    defi_agent = await create_defi_agent(model)
    websearch_agent=await create_websearch_agent(model)

    # ---------------------------------------------------------
    # Supervisor Node: Task Queue Manager & Domain Guardrail
    # ---------------------------------------------------------
    async def supervisor_node(state: AgentState):
        """
        State Machine Logic:
        1. Parse user request and decompose into atomic TaskItems.
        2. Validate if query falls under supported domain capabilities.
        3. If out-of-scope, route immediately to final synthesis node with rejection message.
        4. Execute queue items sequentially, skipping empty or dummy N/A tasks.
        5. Route to 'finish' (final node) when task queue is exhausted.
        """
        task_queue: List[TaskItem] = list(state.get("task_queue") or [])

        # PHASE 1: INITIAL DECOMPOSITION & GUARDRAIL CHECK (Turn 1)
        if not task_queue and state.get("messages"):
            user_request = state["messages"][0].content
            logger.info("=" * 70)
            logger.info("SUPERVISOR: TASK DECOMPOSITION PHASE")
            logger.info(f"Analyzing User Request: '{user_request}'")

            planner_prompt = SystemMessage(
                content="""
You are an expert Task Planning Agent coordinating SPECIALIST sub-agents ONLY.

Supported Domains:
1. 'travel'     : Flight searches, itineraries, location travel info.
2. 'currency'   : Exchange rates, monetary conversions.
3. 'worldclock' : Current time, time zone calculations.
4. 'defi'       : Crypto liquidity pools, risk/entry/exit signals.
5. 'websearch'  : Web searches for real-time news, facts, and current information, content extraction from specific URLs.

STRICT DOMAIN BOUNDARY RULES:
- If the user request does NOT fall into one of the 4 supported domains above (e.g., phone password resets, general tech support, general coding, recipes):
  * Do NOT assign it to any agent.
  * If the schema includes `is_supported`, set `is_supported` to False and state the `rejection_reason`.
  * Return an empty list for `tasks`.
- NEVER assign unrelated topics to a specialist (e.g., NEVER route tech/password support to 'travel').
- NEVER generate dummy, filler, or 'N/A' sub-tasks.
- Write each valid 'query' as a clear, self-contained instruction.
"""
            )

            plan: InitialPlan = await planner_model.ainvoke(
                [planner_prompt, state["messages"][0]]
            )

            # Extract fields safely regardless of schema variation
            is_supported = getattr(plan, "is_supported", True)
            rejection_reason = getattr(plan, "rejection_reason", None)

            # Filter valid, non-empty, non-N/A subtasks
            valid_tasks: List[TaskItem] = []
            if hasattr(plan, "tasks") and plan.tasks:
                for item in plan.tasks:
                    query_str = item.query.strip() if item.query else ""
                    if query_str and query_str.upper() not in ["N/A", "NONE", "NULL"]:
                        valid_tasks.append(
                            {
                                "agent": item.agent.strip().lower(),
                                "query": query_str,
                                "completed": False,
                            }
                        )

            # GUARDRAIL TRIGGER: Unsupported Domain or Empty Task Queue
            if not is_supported or not valid_tasks:
                reason = (
                    rejection_reason
                    or "Request is outside the supported domains (Travel, Currency, World Clock, DeFi)."
                )
                logger.warning(f"[DOMAIN GUARDRAIL] Out-of-Scope Query Detected: {reason}")
                return {
                    "next_agent": "finish",
                    "current_sub_task": None,
                    "task_queue": [],
                    "messages": [AIMessage(content=f"OUT_OF_SCOPE: {reason}")],
                }

            task_queue = valid_tasks
            logger.info(f"Generated Task Queue ({len(task_queue)} sub-tasks):")
            for idx, task in enumerate(task_queue, 1):
                logger.info(f"Task {idx}: [{task['agent'].upper()}] -> {task['query']}")
            logger.info("=" * 70)

        # PHASE 2: QUEUE SELECTION & ROUTING
        next_task: Optional[TaskItem] = None
        for task in task_queue:
            if not task["completed"]:
                next_task = task
                break

        if next_task:
            next_task["completed"] = True
            next_agent = next_task["agent"]
            current_sub_task = next_task["query"]

            completed_count = sum(1 for t in task_queue if t["completed"])
            logger.info("SUPERVISOR ROUTING DECISION")
            logger.info(f"Progress         : {completed_count}/{len(task_queue)} tasks executed")
            logger.info(f"Routing To       : {next_agent}")
            logger.info(f"Dispatching Query: {current_sub_task}")
        else:
            next_agent = "finish"
            current_sub_task = None
            logger.info("SUPERVISOR ROUTING DECISION")
            logger.info("Task Queue Exhausted. All sub-tasks completed.")
            logger.info("Routing To      : final (Synthesis Node)")

        return {
            "next_agent": next_agent,
            "current_sub_task": current_sub_task,
            "task_queue": task_queue,
        }

    # ---------------------------------------------------------
    # Specialist Nodes (Isolated Sub-Task Execution)
    # ---------------------------------------------------------
    async def travel_node(state: AgentState):
        sub_task = state["current_sub_task"]
        logger.info(f"Executing: {sub_task}")

        result = await travel_agent.ainvoke({"messages": [HumanMessage(content=sub_task)]})
        content = result["messages"][-1].content

        return {
            "messages": [AIMessage(content=f"Sub-Task Result ({sub_task}):\n{content}")],
        }

    async def currency_node(state: AgentState):
        sub_task = state["current_sub_task"]
        logger.info(f"Executing: {sub_task}")

        result = await currency_agent.ainvoke({"messages": [HumanMessage(content=sub_task)]})
        content = result["messages"][-1].content

        return {
            "messages": [AIMessage(content=f"Sub-Task Result ({sub_task}):\n{content}")],
        }

    async def worldclock_node(state: AgentState):
        sub_task = state["current_sub_task"]
        logger.info(f"Executing: {sub_task}")

        result = await worldclock_agent.ainvoke({"messages": [HumanMessage(content=sub_task)]})
        content = result["messages"][-1].content
        logger.info(f"Complete. Response length: {len(content)} chars")

        return {
            "messages": [AIMessage(content=f"Sub-Task Result ({sub_task}):\n{content}")],
        }

    async def defi_node(state: AgentState):
        sub_task = state["current_sub_task"]
        logger.info(f"Executing: {sub_task}")

        result = await defi_agent.ainvoke({"messages": [HumanMessage(content=sub_task)]})
        content = result["messages"][-1].content

        return {
            "messages": [AIMessage(content=f"Sub-Task Result ({sub_task}):\n{content}")],
        }


    async def websearch_node(state:AgentState):
        sub_task = state["current_sub_task"]
        logger.info(f"Executing: {sub_task}")

        result = await websearch_agent.ainvoke({"messages": [HumanMessage(content=sub_task)]})
        content = result["messages"][-1].content

        return {
            "messages": [AIMessage(content=f"Sub-Task Result ({sub_task}):\n{content}")],
        }


    # ---------------------------------------------------------
    # Final Synthesis Node (Handles Normal Output & Refusals)
    # ---------------------------------------------------------
    async def final_node(state: AgentState):
        logger.info("Compiling final unified output for user...")
# Guardrail Refusal Handler
        last_msg = state["messages"][-1].content if state.get("messages") else ""
        if isinstance(last_msg, str) and "OUT_OF_SCOPE:" in last_msg:
            refusal_response = (
                "I am sorry, but I cannot assist with that request. "
                "I am a specialized assistant trained exclusively for **Travel**, **Currency Conversions**, "
                "**World Clock**, and **DeFi Metrics**."
            )
            return {"messages": [AIMessage(content=refusal_response)]}

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
        logger.info("Final response successfully generated.")
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
    workflow.add_node("websearch", websearch_node)
    workflow.add_node("final", final_node)

    # Set Entry Point
    workflow.add_edge(START, "supervisor")

    # Conditional Routing based on supervisor's next_agent
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        AgentMap,
    )

    # Specialist nodes loop back to supervisor to process remaining queue items
    workflow.add_edge("travel", "supervisor")
    workflow.add_edge("currency", "supervisor")
    workflow.add_edge("worldclock", "supervisor")
    workflow.add_edge("defi", "supervisor")
    workflow.add_edge("websearch","supervisor")

    # Final Node terminates execution
    workflow.add_edge("final", END)

    return workflow.compile()