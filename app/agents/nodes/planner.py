import logfire

from app.agents.state import AgentState
from app.gateway import LlmTier, get_chat_llm


# Portkey-backed LLM with the standard .invoke() interface.
llm = get_chat_llm(
    LlmTier.PRIMARY,
    feature="planner",
    temperature=0,
)


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.

    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    with logfire.span(
        "[Planner] Select route",
        message_count=len(state["messages"]),
        history_length=len(history),
        user_message_length=len(user_message),
    ):
        decision = llm.invoke(prompt).content.strip()
        route = "conversational" if decision == "CONVERSATIONAL" else "retrieval"
        logfire.info(
            "[Planner] Route selected",
            route=route,
            search_query=None if route == "conversational" else decision,
        )

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"],
    }
