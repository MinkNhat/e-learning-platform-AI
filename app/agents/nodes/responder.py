import logfire

from app.agents.state import AgentState
from app.gateway import LlmTier, extract_cache_status, get_chat_llm


llm = get_chat_llm(
    LlmTier.SECONDARY,
    feature="responder",
    temperature=0.1,
)


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Portkey supplies response headers through the LangChain response metadata so
    the cache status can still be surfaced in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    response_mode = "conversational" if query == "CONVERSATIONAL" else "rag"
    full_context = ""
    included_document_count = 0

    if query == "CONVERSATIONAL":
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        max_context_chars = 25000

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
                included_document_count += 1
            else:
                logfire.warning(
                    "[WARNING][Responder] Context truncated",
                    context_length=len(full_context),
                    max_context_chars=max_context_chars,
                    input_document_count=len(state["documents"]),
                    included_document_count=included_document_count,
                    next_document_length=len(doc),
                )
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span(
        "[Responder] Generate response",
        response_mode=response_mode,
        prompt_length=len(prompt),
        history_length=len(history_str),
        context_length=len(full_context),
        input_document_count=len(state["documents"]),
        included_document_count=included_document_count,
    ):
        response = llm.invoke(prompt)
        content = response.content
        cache_status = extract_cache_status(response)
        is_cache_hit = cache_status == "HIT"

        logfire.info(
            "[Responder] Response generated",
            response_mode=response_mode,
            response_length=len(content),
            cache_status=cache_status,
        )

        if is_cache_hit:
            plan_update = state["plan"] + ["Cache: Hit"]
            status = "Cache hit — instant response."
        else:
            plan_update = state["plan"]
            status = "Response generated."

        return {
            "final_answer": content,
            "status": status,
            "plan": plan_update,
            "messages": [{"role": "assistant", "content": content}],
        }
