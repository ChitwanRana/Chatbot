import streamlit as st
from tools_backend import chatbot, retrieve_all_threads  # ✅ your integrated backend file
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import os 

os.environ['LANGCHAIN_PROJECT']='Tools-Chatbot'

# =========================== Utilities ===========================
def generate_thread_id() -> str:
    """Generate a new unique thread ID (as string)."""
    return str(uuid.uuid4())

def reset_chat():
    """Start a new chat thread."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id: str):
    """Add new thread to session if not already present."""
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id: str):
    """Load saved conversation from SQLite checkpoint."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    if not state or not hasattr(state, "values"):
        return []
    return state.values.get("messages", [])

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads() or []

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("🧵 My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(f"{thread_id[:8]}..."):  # show shorter ID
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                temp_messages.append({"role": "assistant", "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

st.title("💬 LangGraph Chatbot with Tools & RAG")

# Render previous messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Display user message immediately
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Chatbot configuration for this session/thread
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant message streaming block
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            """Stream only assistant messages while updating tool status."""
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)], "query": user_input},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Show when a tool is running
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Running `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Running `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream AI tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Update tool status if used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant response
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message or ""}
    )
