import streamlit as st
from langgraph_backend import chatbot, build_vectorstore, get_uploaded_documents
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# Set page config for better UI
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    
    .upload-container {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    
    .thread-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        background-color: #f1f3f4;
    }
    
    .chat-input-container {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 1rem 0;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.session_state['current_thread_docs'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def load_thread_documents(thread_id):
    """Load documents for a specific thread."""
    try:
        return get_uploaded_documents(str(thread_id))
    except:
        return []

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'current_thread_docs' not in st.session_state:
    st.session_state['current_thread_docs'] = []

if 'show_upload' not in st.session_state:
    st.session_state['show_upload'] = False

add_thread(st.session_state['thread_id'])

# Load documents for current thread
st.session_state['current_thread_docs'] = load_thread_documents(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

with st.sidebar:
    st.title('🤖 RAG Chatbot')
    
    # Current Thread Info
    st.markdown("---")
    st.subheader('📎 Current Thread')
    thread_short_id = str(st.session_state['thread_id'])[:8]
    st.code(f"ID: {thread_short_id}...")
    
    # Document count badge
    doc_count = len(st.session_state['current_thread_docs'])
    if doc_count > 0:
        st.markdown(f'<div class="status-badge status-success">📄 {doc_count} Documents</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-badge status-warning">📄 No Documents</div>', unsafe_allow_html=True)
    
    # Show uploaded documents for current thread
    if st.session_state['current_thread_docs']:
        st.markdown("**Documents in this thread:**")
        for doc in st.session_state['current_thread_docs']:
            st.markdown(f"• {doc}")
    
    # Chat Management
    st.markdown("---")
    st.subheader('💬 Chat Management')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🆕 New Chat', use_container_width=True):
            reset_chat()
            st.rerun()
    
    with col2:
        if st.button('📎 Upload Docs', use_container_width=True):
            st.session_state['show_upload'] = not st.session_state['show_upload']
            st.rerun()
    
    # Thread History
    st.markdown("---")
    st.subheader('📝 Recent Conversations')
    
    for thread_id in st.session_state['chat_threads'][::-1][:5]:  # Show last 5 threads
        thread_docs = load_thread_documents(thread_id)
        doc_count = len(thread_docs)
        
        # Create thread display
        thread_short = str(thread_id)[:8]
        
        if thread_id == st.session_state['thread_id']:
            st.markdown(f'<div class="thread-item" style="background-color: #e3f2fd;"><strong>🔵 {thread_short}... ({doc_count} docs)</strong></div>', unsafe_allow_html=True)
        else:
            if st.button(f"💬 {thread_short}... ({doc_count} docs)", key=f"thread_{thread_id}", use_container_width=True):
                st.session_state['thread_id'] = thread_id
                messages = load_conversation(thread_id)

                temp_messages = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        role='user'
                    else:
                        role='assistant'
                    temp_messages.append({'role': role, 'content': msg.content})

                st.session_state['message_history'] = temp_messages
                st.session_state['current_thread_docs'] = load_thread_documents(thread_id)
                st.rerun()

# **************************************** Main UI ************************************

# Header
st.markdown('<div class="main-header"><h1>🤖 RAG-Enabled Chatbot</h1></div>', unsafe_allow_html=True)

# Status Bar
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    if st.session_state['current_thread_docs']:
        st.success(f"📚 Ready to chat with {len(st.session_state['current_thread_docs'])} documents")
    else:
        st.warning("📁 No documents uploaded. Upload PDFs to enable document chat.")

with col2:
    st.info(f"🔗 Thread: {str(st.session_state['thread_id'])[:8]}...")

with col3:
    if st.button("📎", help="Upload Documents", key="main_upload_btn"):
        st.session_state['show_upload'] = not st.session_state['show_upload']
        st.rerun()

# Upload Section (Collapsible)
if st.session_state['show_upload']:
    st.markdown("---")
    st.subheader("📎 Upload Documents")
    
    upload_col1, upload_col2 = st.columns([3, 1])
    
    with upload_col1:
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents. OCR is supported for scanned documents.",
            key="main_uploader"
        )
    
    with upload_col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        process_btn = st.button('🔄 Process', use_container_width=True, type="primary")
    
    if uploaded_files and process_btn:
        with st.spinner('🔄 Processing documents...'):
            try:
                vectorstore, chunk_count = build_vectorstore(uploaded_files, str(st.session_state['thread_id']))
                if vectorstore:
                    st.session_state['current_thread_docs'] = load_thread_documents(st.session_state['thread_id'])
                    st.success(f'✅ Successfully processed {len(uploaded_files)} documents ({chunk_count} chunks)')
                    st.session_state['show_upload'] = False
                    st.rerun()
                else:
                    st.error('❌ Failed to process documents')
            except Exception as e:
                st.error(f'❌ Error processing documents: {str(e)}')

# Chat History Container
st.markdown("---")
st.subheader("💬 Conversation")

# Chat messages container with custom styling
chat_container = st.container()
with chat_container:
    if st.session_state['message_history']:
        for message in st.session_state['message_history']:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
    else:
        st.markdown("👋 **Welcome!** Start a conversation by typing a message below.")
        if not st.session_state['current_thread_docs']:
            st.markdown("💡 **Tip:** Upload PDF documents to chat with them using the upload button above.")

# **************************************** Chat Input (Bottom) ************************************

st.markdown("---")
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)

# Chat input section
input_col1, input_col2 = st.columns([6, 1])

with input_col1:
    user_input = st.chat_input('💬 Type your message here...', key="main_chat_input")

with input_col2:
    # Quick upload toggle
    st.write("")  # Add spacing for alignment
    if st.button("📎", help="Quick Upload", key="quick_upload_toggle", use_container_width=True):
        st.session_state['show_upload'] = not st.session_state['show_upload']
        st.rerun()

# Quick upload (appears when toggled)
if st.session_state.get('show_quick_upload', False):
    st.write("**Quick Upload:**")
    quick_files = st.file_uploader(
        "Drop PDFs here",
        type=['pdf'],
        accept_multiple_files=True,
        key="quick_uploader",
        label_visibility="collapsed"
    )
    
    if quick_files:
        if st.button('⚡ Quick Process', key="quick_process"):
            with st.spinner('⚡ Processing...'):
                try:
                    vectorstore, chunk_count = build_vectorstore(quick_files, str(st.session_state['thread_id']))
                    if vectorstore:
                        st.session_state['current_thread_docs'] = load_thread_documents(st.session_state['thread_id'])
                        st.success(f'✅ Added {len(quick_files)} documents')
                        st.session_state['show_quick_upload'] = False
                        st.rerun()
                except Exception as e:
                    st.error(f'❌ Error: {str(e)}')

st.markdown('</div>', unsafe_allow_html=True)

# Handle user input
if user_input:
    # Add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    # Show user message immediately
    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': str(st.session_state['thread_id'])}}

    # Generate AI response
    with st.chat_message("assistant"):
        try:
            def ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "query": user_input,
                        "thread_id": str(st.session_state['thread_id'])
                    },
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())
            st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
            
        except Exception as e:
            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state['message_history'].append({'role': 'assistant', 'content': error_msg})

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: gray; font-size: 0.8rem;">🤖 RAG-Enabled Chatbot with OCR Support</div>',
    unsafe_allow_html=True
)