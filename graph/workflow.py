# type:ignore
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
from graph.models import AgentMap, InitialPlan
from graph.state import AgentState, TaskItem

logger = logging.getLogger(__name__)

# Two retries after the initial attempt = three total attempts.
MAX_RETRIES = 2
MAX_ATTEMPTS = MAX_RETRIES + 1

SUPPORTED_AGENTS = {
    "travel",
    "currency",
    "worldclock",
    "defi",
    "websearch",
}


async def build_workflow():
    model = ChatOpenAI(model=MODEL, temperature=0.0)
    planner_model = model.with_structured_output(InitialPlan)

    # Initialize specialist agents once.
    travel_agent = await create_travel_agent(model)
    currency_agent = await create_currency_agent(model)
    worldclock_agent = await create_worldclock_agent(model)
    defi_agent = await create_defi_agent(model)
    websearch_agent = await create_websearch_agent(model)

    agents = {
        "travel": travel_agent,
        "currency": currency_agent,
        "worldclock": worldclock_agent,
        "defi": defi_agent,
        "websearch": websearch_agent,
    }

    # ---------------------------------------------------------
    # Supervisor: planning, guardrails, and task selection
    # ---------------------------------------------------------

    async def supervisor_node(state: AgentState):
        task_queue: List[TaskItem] = [
            dict(task) for task in (state.get("task_queue") or [])
        ]

        # PHASE 1: Plan the current user request.
        if not task_queue and state.get("messages"):
            messages = state.get("messages", [])

            latest_user_message = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, HumanMessage)
                ),
                None,
            )

            if latest_user_message is None:
                raise ValueError("No user message found in the conversation.")

            user_query = latest_user_message.content

            if not isinstance(user_query, str):
                raise ValueError("The current user request must be text.")

            logger.info("=" * 70)
            logger.info("SUPERVISOR: TASK DECOMPOSITION")
            logger.info("Current user request: %s", user_query)

            # Include earlier conversation only as context.
            # The planner must create tasks for the latest request only.
            previous_messages = messages[:messages.index(latest_user_message)]

            context_lines = []

            for message in previous_messages[-10:]:
                if isinstance(message, HumanMessage):
                    role = "User"
                elif isinstance(message, AIMessage):
                    role = "Assistant"
                else:
                    continue

                content = message.content
                if isinstance(content, str) and content.strip():
                    context_lines.append(f"{role}: {content}")

            conversation_context = (
                "\n".join(context_lines)
                if context_lines
                else "No previous conversation."
            )

            planner_prompt = SystemMessage(
                content="""
    You are an expert task-planning agent coordinating specialist sub-agents.

    SUPPORTED DOMAINS:
    1. travel: Flight searches, itineraries, and travel information.
    2. currency: Exchange rates and monetary conversions.
    3. worldclock: Current time and time-zone calculations.
    4. defi: Crypto liquidity pools and risk/entry/exit signals.
    5. websearch: Current news, facts, web searches, and URL extraction.

    RULES:
    - All five domains are supported.
    - A request may require one or multiple specialist agents.
    - Reject requests outside these supported domains.
    - Never reject a request merely because it requires multiple agents.
    - Never create dummy, filler, or N/A tasks.
    - Every task must have a valid agent name and a clear query.
    - Valid agent names are exactly: travel, currency, worldclock, defi, websearch.

    CONVERSATION RULES:
    - The latest user request is the task to execute now.
    - Use previous conversation only to resolve references such as "there",
    "that flight", "its price", or "convert it".
    - Do not repeat tasks from previous turns unless explicitly requested.
    - For follow-up requests, include relevant details from the conversation
    in the specialist query when possible.
    - Do not assume previous results are current for time-sensitive requests.
    - Set is_supported=True and create tasks for supported requests.
    """
            )

            planner_input = [
                planner_prompt,
                HumanMessage(
                    content=(
                        f"PREVIOUS CONVERSATION CONTEXT:\n"
                        f"{conversation_context}\n\n"
                        f"CURRENT USER REQUEST:\n{user_query}"
                    )
                ),
            ]

            plan: InitialPlan = await planner_model.ainvoke(planner_input)

            logger.info("Planner output: %s", plan.model_dump())

            is_supported = getattr(plan, "is_supported", True)
            rejection_reason = getattr(plan, "rejection_reason", None)

            valid_tasks: List[TaskItem] = []

            for item in getattr(plan, "tasks", []) or []:
                query = (item.query or "").strip()
                agent_name = (item.agent or "").strip().lower()

                if not query or query.upper() in {"N/A", "NONE", "NULL"}:
                    continue

                if agent_name not in SUPPORTED_AGENTS:
                    logger.warning(
                        "Ignoring task with unsupported agent: %s",
                        agent_name,
                    )
                    continue

                valid_tasks.append(
                    {
                        "agent": agent_name,
                        "query": query,
                        "status": "pending",
                        "attempts": 0,
                        "error": None,
                    }
                )

            # Reject unsupported requests without invoking specialists.
            if not is_supported or not valid_tasks:
                reason = rejection_reason or (
                    "The request is outside the supported domains: "
                    "travel, currency, world clock, DeFi, and web search."
                )

                logger.warning("Domain guardrail triggered: %s", reason)

                return {
                    "next_agent": "finish",
                    "current_sub_task": None,
                    "task_queue": [],
                    "messages": [
                        AIMessage(content=f"OUT_OF_SCOPE: {reason}")
                    ],
                }

            task_queue = valid_tasks

            logger.info("Created %s task(s)", len(task_queue))

            for index, task in enumerate(task_queue, start=1):
                logger.info(
                    "Task %s: [%s] %s",
                    index,
                    task["agent"].upper(),
                    task["query"],
                )

        # PHASE 2: Select the next pending task.
        next_task: Optional[TaskItem] = next(
            (
                task
                for task in task_queue
                if task["status"] == "pending"
            ),
            None,
        )

        if next_task is not None:
            next_task["status"] = "running"

            next_agent = next_task["agent"]
            current_sub_task = next_task["query"]

            logger.info(
                "Routing to %s | Attempt %s/%s",
                next_agent,
                next_task["attempts"] + 1,
                MAX_ATTEMPTS,
            )
        else:
            next_agent = "finish"
            current_sub_task = None

            logger.info("No pending tasks remain. Starting final synthesis.")

        return {
            "next_agent": next_agent,
            "current_sub_task": current_sub_task,
            "task_queue": task_queue,
        }

    # ---------------------------------------------------------
    # Shared specialist executor: bounded retries and failures
    # ---------------------------------------------------------

    async def execute_specialist(
        state: AgentState,
        agent,
        agent_name: str,
    ):
        sub_task = state.get("current_sub_task")

        task_queue: List[TaskItem] = [
            dict(task) for task in state["task_queue"]
        ]

        current_task = next(
            (
                task
                for task in task_queue
                if task["query"] == sub_task
                and task["agent"] == agent_name
                and task["status"] == "running"
            ),
            None,
        )

        if current_task is None:
            raise RuntimeError(
                f"No running task found for agent '{agent_name}'."
            )

        # Count the attempt before invoking the specialist.
        current_task["attempts"] += 1
        attempt = current_task["attempts"]

        try:
            logger.info(
                "Executing %s | Attempt %s/%s",
                agent_name,
                attempt,
                MAX_ATTEMPTS,
            )

            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=sub_task)
                    ]
                }
            )

            result_messages = result.get("messages", [])

            if not result_messages:
                raise ValueError(
                    "The specialist returned no messages."
                )

            content = result_messages[-1].content

            # Success: mark the task completed.
            current_task["status"] = "completed"
            current_task["error"] = None

            logger.info(
                "Agent %s completed successfully on attempt %s.",
                agent_name,
                attempt,
            )

            return {
                "task_queue": task_queue,
                "messages": [
                    AIMessage(
                        content=(
                            f"Sub-Task Result ({sub_task}):\n"
                            f"{content}"
                        )
                    )
                ],
            }

        except Exception as exc:
            error_message = (
                f"{type(exc).__name__}: {str(exc)}"
            )[:1000]

            current_task["error"] = error_message

            if attempt < MAX_ATTEMPTS:
                # Return the task to pending so the supervisor
                # selects it again.
                current_task["status"] = "pending"

                logger.warning(
                    "Agent %s failed on attempt %s/%s. "
                    "Retrying. Error: %s",
                    agent_name,
                    attempt,
                    MAX_ATTEMPTS,
                    error_message,
                )

                # Do not add intermediate failures to conversation
                # history. The next attempt will try the task again.
                return {"task_queue": task_queue}

            # Retry limit reached: permanently fail this task.
            current_task["status"] = "failed"

            logger.error(
                "Agent %s exhausted all %s attempts. "
                "Skipping task. Error: %s",
                agent_name,
                MAX_ATTEMPTS,
                error_message,
            )

            # Keep the error in task_queue for final synthesis.
            # The supervisor will select the next pending task.
            return {"task_queue": task_queue}

    # ---------------------------------------------------------
    # Specialist nodes
    # ---------------------------------------------------------
    async def travel_node(state: AgentState):
        return await execute_specialist(
            state, travel_agent, "travel"
        )

    async def currency_node(state: AgentState):
        return await execute_specialist(
            state, currency_agent, "currency"
        )

    async def worldclock_node(state: AgentState):
        return await execute_specialist(
            state, worldclock_agent, "worldclock"
        )

    async def defi_node(state: AgentState):
        return await execute_specialist(
            state, defi_agent, "defi"
        )

    async def websearch_node(state: AgentState):
        return await execute_specialist(
            state, websearch_agent, "websearch"
        )

    # ---------------------------------------------------------
    # Final synthesis
    # ---------------------------------------------------------
    async def final_node(state: AgentState):
        logger.info("Compiling final unified response")

        messages = state.get("messages", [])
        last_message = messages[-1].content if messages else ""

        # Handle unsupported requests without normal synthesis.
        if (
            isinstance(last_message, str)
            and "OUT_OF_SCOPE:" in last_message
        ):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I cannot assist with that request. "
                            "I specialize exclusively in travel, "
                            "currency conversions, world clock, "
                            "DeFi metrics, and web search."
                        )
                    )
                ]
            }

        failed_tasks = [
            task
            for task in (state.get("task_queue") or [])
            if task["status"] == "failed"
        ]

        if failed_tasks:
            failure_lines = [
                (
                    f"- Task: {task['query']}\n"
                    f"  Attempts: {task['attempts']}\n"
                    f"  Last error: {task['error'] or 'Unknown error'}"
                )
                for task in failed_tasks
            ]

            failure_context = SystemMessage(
                content=(
                    "Some tasks could not be completed after the "
                    f"maximum of {MAX_ATTEMPTS} attempts.\n"
                    "Report these failures honestly, without "
                    "inventing results:\n"
                    + "\n".join(failure_lines)
                )
            )
        else:
            failure_context = SystemMessage(
                content="All planned tasks completed successfully."
            )

        final_prompt = SystemMessage(
            content="""
You are the final response agent.

Synthesize the successful specialist results into one clear,
well-structured answer to the user's original request.

Rules:
- Answer every part for which results are available.
- Preserve successful results even if other tasks failed.
- Clearly identify incomplete parts of the request.
- Never invent results for failed tasks.
- If every task failed, explain that the requested information
  could not be retrieved.
- Do not expose internal agent names, retry counters, or
  orchestration mechanics.
- Present the response naturally using headings or bullet points.
"""
        )

        response = await model.ainvoke(
            [
                final_prompt,
                *messages,
                failure_context,
            ]
        )

        logger.info("Final response generated successfully.")

        return {"messages": [response]}

    # ---------------------------------------------------------
    # Graph construction
    # ---------------------------------------------------------
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("travel", travel_node)
    workflow.add_node("currency", currency_node)
    workflow.add_node("worldclock", worldclock_node)
    workflow.add_node("defi", defi_node)
    workflow.add_node("websearch", websearch_node)
    workflow.add_node("final", final_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        AgentMap,
    )

    # Every specialist returns control to the supervisor.
    workflow.add_edge("travel", "supervisor")
    workflow.add_edge("currency", "supervisor")
    workflow.add_edge("worldclock", "supervisor")
    workflow.add_edge("defi", "supervisor")
    workflow.add_edge("websearch", "supervisor")

    workflow.add_edge("final", END)

    return workflow.compile()
