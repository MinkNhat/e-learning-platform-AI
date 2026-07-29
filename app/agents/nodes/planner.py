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
    Determine whether the learner's request needs learning-material retrieval.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are the routing planner for an e-learning RAG assistant.
    Analyze the conversation history and the learner's latest message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. Output 'CONVERSATIONAL' only for greetings, casual conversation, or a
       request that can be answered fully from the conversation history without
       consulting learning materials.
    2. For a question about a lesson, concept, example, exercise, assignment,
       exam preparation, or indexed learning material, output one concise,
       standalone retrieval query. Resolve references from the history and keep
       important course or topic terms from the learner's wording.
    3. Do not answer the learner and do not explain your decision.

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
        return {"current_query": "CONVERSATIONAL"}

    return {"current_query": decision}
