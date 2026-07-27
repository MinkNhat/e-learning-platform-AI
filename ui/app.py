import os
import time
import uuid

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv

TYPEWRITER_DELAY_SECONDS = 0.005


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(
        token=token,
        service_name="rag-ui",
    )
    # Requests instrumentation is disabled because of an OpenTelemetry issue on Windows.
except Exception as error:
    print(f"Logfire Init Error in UI: {error}")


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] {
        width: 100%;
        padding: 0.5rem 0;
        gap: 0.65rem;
        background: transparent !important;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarAssistant"]
    ) {
        margin-right: auto;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
        width: min(82%, 52rem);
        margin-left: auto;
        flex-direction: row-reverse;
    }

    [data-testid="stChatMessageContent"] {
        text-align: left;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) [data-testid="stChatMessageContent"] {
        max-width: calc(100% - 3rem);
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        background-color: rgba(105, 133, 169, 0.14);
        border: 1px solid rgba(105, 133, 169, 0.24);
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        color: #35564a !important;
        background-color: #d8e8e1 !important;
        border: 1px solid #bfd4cb;
    }

    [data-testid="stChatMessageAvatarUser"] {
        color: #3e5874 !important;
        background-color: #dce6f2 !important;
        border: 1px solid #c4d3e4;
    }

    @media (max-width: 640px) {
        [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ) {
            width: 94%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(
        "[UI] Session created",
        session_id=st.session_state.session_id,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


def normalize_markdown(markdown: str) -> str:
    """Align fenced code blocks so CommonMark does not split them incorrectly."""
    normalized_lines: list[str] = []
    active_fence: tuple[str, int] | None = None

    for line in markdown.splitlines(keepends=True):
        stripped_line = line.lstrip(" \t")
        marker_character = stripped_line[0] if stripped_line else ""
        marker_length = (
            len(stripped_line) - len(stripped_line.lstrip(marker_character))
            if marker_character in ("`", "~")
            else 0
        )

        if active_fence is None and marker_length >= 3:
            active_fence = (marker_character, marker_length)
            line = stripped_line
        elif active_fence is not None:
            fence_character, fence_length = active_fence
            if (
                marker_character == fence_character
                and marker_length >= fence_length
                and stripped_line[marker_length:].strip() == ""
            ):
                active_fence = None
                line = stripped_line

        normalized_lines.append(line)

    if active_fence is not None:
        fence_character, fence_length = active_fence
        if normalized_lines and not normalized_lines[-1].endswith("\n"):
            normalized_lines[-1] += "\n"
        normalized_lines.append(fence_character * fence_length)

    return "".join(normalized_lines)


def render_message_details(
    details_placeholder,
    context_placeholder,
    message: dict,
) -> None:
    """Render optional assistant metadata above the answer."""
    steps = message.get("steps", [])
    if steps:
        with details_placeholder.container():
            with st.expander("Processing details"):
                for step in steps:
                    st.write(step)

    sources = message.get("sources", [])
    if sources:
        with context_placeholder.container():
            with st.expander("Retrieved context"):
                for index, source in enumerate(sources, start=1):
                    with st.expander(f"Chunk {index}"):
                        st.write(source)


def render_answer(placeholder, message: dict) -> None:
    content = normalize_markdown(message["content"])
    with placeholder.container():
        if message.get("error"):
            st.error(content)
        else:
            st.markdown(content)


def render_message(message: dict) -> None:
    """Render stored messages with the same structure on every rerun."""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            details_placeholder = st.empty()
            context_placeholder = st.empty()
            answer_placeholder = st.empty()
            render_message_details(
                details_placeholder,
                context_placeholder,
                message,
            )
            render_answer(answer_placeholder, message)
        else:
            st.markdown(message["content"])


def render_typewriter(placeholder, answer: str) -> None:
    """Render an answer one character at a time into a stable UI slot."""
    def stream_characters():
        for character in answer:
            yield character
            time.sleep(TYPEWRITER_DELAY_SECONDS)

    with placeholder.container():
        st.write_stream(stream_characters(), cursor="▌")


# --- SIDEBAR ---
with st.sidebar:
    st.title("Agentic Assistant")
    st.markdown("---")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    if st.button("Clear history and memory", width="stretch", type="primary"):
        previous_session_id = st.session_state.session_id
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        logfire.info(
            "[UI] Session reset",
            previous_session_id=previous_session_id,
            session_id=st.session_state.session_id,
        )
        st.rerun()

# --- MAIN CHAT ---
st.title("Enterprise Agentic Assistant")


# Display history
for message in st.session_state.messages:
    render_message(message)

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with logfire.span(
        "[UI] Process chat",
        session_id=st.session_state.session_id,
        prompt_preview=prompt[:200],
        prompt_length=len(prompt),
        message_count=len(st.session_state.messages),
    ) as chat_span:
        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        url = f"{base_url}/query"

        with st.chat_message("assistant"):
            details_placeholder = st.empty()
            context_placeholder = st.empty()
            answer_placeholder = st.empty()
            try:
                with answer_placeholder.container():
                    with st.spinner("Generating answer..."):
                        with logfire.span(
                            "[UI] Call backend",
                            backend_url=url,
                            timeout_seconds=60,
                        ) as backend_span:
                            payload = {
                                "q": prompt,
                                "thread_id": st.session_state.session_id,
                            }
                            response = requests.post(url, json=payload, timeout=60)
                            data = response.json()
                            backend_status = data.get("status")
                            backend_failed = (
                                response.status_code >= 400
                                or backend_status == "error"
                            )
                            backend_span.set_attributes(
                                {
                                    "http_status_code": response.status_code,
                                    "response_size": len(response.content),
                                    "backend_status": backend_status,
                                    "outcome": (
                                        "error" if backend_failed else "completed"
                                    ),
                                }
                            )
                            if backend_failed:
                                backend_span.set_level("error")
                                logfire.error(
                                    "[ERROR][UI] Backend returned an error",
                                    backend_url=url,
                                    http_status_code=response.status_code,
                                    backend_status=backend_status,
                                    session_id=st.session_state.session_id,
                                )
            except Exception as error:
                chat_span.set_attribute("outcome", "error")
                chat_span.set_level("error")
                logfire.exception(
                    "[ERROR][UI] Backend request failed",
                    error=str(error),
                    error_type=type(error).__name__,
                    backend_url=url,
                    session_id=st.session_state.session_id,
                )
                assistant_message = {
                    "role": "assistant",
                    "content": "Backend unavailable.",
                    "error": True,
                }
                st.session_state.messages.append(assistant_message)
                render_answer(answer_placeholder, assistant_message)
            else:
                steps = data.get("thought_process") or []
                sources = data.get("sources") or []
                full_answer = normalize_markdown(
                    data.get("answer") or "No response."
                )
                assistant_message = {
                    "role": "assistant",
                    "content": full_answer,
                    "steps": steps,
                    "sources": sources,
                }
                st.session_state.messages.append(assistant_message)
                render_message_details(
                    details_placeholder,
                    context_placeholder,
                    assistant_message,
                )
                render_typewriter(answer_placeholder, full_answer)

                if backend_failed:
                    chat_span.set_level("error")
                chat_span.set_attributes(
                    {
                        "outcome": "error_response" if backend_failed else "completed",
                        "answer_length": len(full_answer),
                        "source_count": len(sources),
                        "reasoning_step_count": len(steps),
                    }
                )
                logfire.info(
                    "[UI] Chat completed",
                    session_id=st.session_state.session_id,
                    answer_length=len(full_answer),
                    source_count=len(sources),
                    reasoning_step_count=len(steps),
                    message_count=len(st.session_state.messages),
                )
