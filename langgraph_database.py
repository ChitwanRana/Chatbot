from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver  # ✅ SQLite Checkpointer
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import sqlite3

# =====================================================================
# ✅ Load environment variables
# =====================================================================
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)

# ✅ Azure credentials
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

if not all([azure_api_key, azure_endpoint, azure_deployment]):
    raise ValueError("❌ Missing Azure OpenAI credentials!")

# =====================================================================
# ✅ Initialize Azure LLM and Embeddings
# =====================================================================
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    deployment_name=azure_deployment,
    api_version=azure_api_version,
)

azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=azure_api_key,
    openai_api_base=azure_endpoint,
    openai_api_type="azure",
    deployment=azure_embedding_deployment,
)

# =====================================================================
# Document Store + RAG Setup
# =====================================================================
VECTOR_STORE_PATH = "vectorstore/faiss_index"

if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

def build_vectorstore(pdf_path):
    """Loads PDF, splits, and builds FAISS vector store."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return vectorstore

def load_vectorstore():
    """Load existing FAISS vector store."""
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# =====================================================================
# Define LangGraph State
# =====================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    retrieved_context: str

# =====================================================================
# RAG Nodes
# =====================================================================
def retrieve_node(state: ChatState):
    """Retrieve relevant context from documents."""
    query = state['query']
    if os.path.exists(VECTOR_STORE_PATH):
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
    else:
        context = "(No document uploaded yet)"
    return {"retrieved_context": context}

def chat_node(state: ChatState):
    """Generate final answer using LLM + context."""
    messages = state["messages"]
    context = state.get("retrieved_context", "")
    user_msg = messages[-1].content

    system_prompt = f"You are a helpful assistant. Use the following context to answer accurately:\n\n{context}\n\nQuestion: {user_msg}"

    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [response]}

# =====================================================================
# ✅ SQLite Checkpointer Integration
# =====================================================================
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# =====================================================================
# Graph Setup
# =====================================================================
graph = StateGraph(ChatState)
graph.add_node("retriever", retrieve_node)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "retriever")
graph.add_edge("retriever", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# =====================================================================
# ✅ Helper Function: Retrieve All Threads
# =====================================================================
def retrieve_all_threads():
    """Fetch all unique thread IDs from SQLite checkpoint database."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
