from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage
import json

# Safe import for PyMuPDF
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("⚠️ PyMuPDF not available. OCR features will be limited.")

# Safe import for OCR
try:
    import pytesseract
    from PIL import Image
    import io
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️ OCR dependencies not available. Scanned PDF support will be limited.")

import tempfile
from langchain_core.documents import Document

# ✅ Load environment variables
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)

# ✅ Azure credentials with better error handling
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

print(f"🔍 Azure Endpoint: {azure_endpoint}")
print(f"🔍 Azure Deployment: {azure_deployment}")
print(f"🔍 API Version: {azure_api_version}")

if not all([azure_api_key, azure_endpoint, azure_deployment]):
    raise ValueError("❌ Missing Azure OpenAI credentials!")

# ✅ Initialize Azure LLM with corrected configuration
llm = AzureChatOpenAI(
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_version=azure_api_version,
    temperature=0.7,
)

# ✅ Initialize embeddings with CORRECT Azure configuration
azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

if azure_embedding_deployment:
    # Use AzureOpenAIEmbeddings instead of OpenAIEmbeddings
    embeddings = AzureOpenAIEmbeddings(
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_embedding_deployment,
        api_version=azure_api_version,
    )
    print(f"✅ Using Azure embeddings: {azure_embedding_deployment}")
else:
    raise ValueError("❌ No Azure embedding deployment found!")

# =====================================================================
# Per-Thread Document Store Setup
# =====================================================================
VECTOR_STORE_BASE_PATH = "vectorstore"

if not os.path.exists(VECTOR_STORE_BASE_PATH):
    os.makedirs(VECTOR_STORE_BASE_PATH)

def get_thread_vector_path(thread_id):
    """Get vector store path for specific thread."""
    return os.path.join(VECTOR_STORE_BASE_PATH, f"thread_{thread_id}")

def get_thread_metadata_path(thread_id):
    """Get metadata file path for specific thread."""
    return os.path.join(VECTOR_STORE_BASE_PATH, f"thread_{thread_id}_metadata.json")

def save_thread_metadata(thread_id, filenames):
    """Save uploaded document names for a thread."""
    metadata_path = get_thread_metadata_path(thread_id)
    
    # Load existing metadata or create new
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            existing_files = json.load(f)
    else:
        existing_files = []
    
    # Add new files (avoid duplicates)
    for filename in filenames:
        if filename not in existing_files:
            existing_files.append(filename)
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(existing_files, f)

def get_thread_documents(thread_id):
    """Get list of uploaded documents for a thread."""
    metadata_path = get_thread_metadata_path(thread_id)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return []

def extract_text_with_ocr(pdf_path):
    """Extract text from PDF using OCR for scanned documents."""
    documents = []
    
    try:
        # Try regular PDF text extraction first
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Check if text was extracted successfully
        total_text = "".join([doc.page_content for doc in docs])
        
        if len(total_text.strip()) > 100:  # If sufficient text found
            return docs
        else:
            # Fallback to OCR if available
            if HAS_PYMUPDF and HAS_OCR:
                print("⚠️ Little text found, using OCR...")
                return extract_with_ocr_fallback(pdf_path)
            else:
                print("⚠️ OCR not available, using extracted text anyway...")
                return docs
            
    except Exception as e:
        print(f"⚠️ PDF extraction failed: {e}")
        if HAS_PYMUPDF and HAS_OCR:
            return extract_with_ocr_fallback(pdf_path)
        else:
            return []

def extract_with_ocr_fallback(pdf_path):
    """Use OCR to extract text from scanned PDFs."""
    if not (HAS_PYMUPDF and HAS_OCR):
        print("❌ OCR dependencies not available")
        return []
        
    documents = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Convert page to image
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            
            if text.strip():  # Only add if text found
                doc = Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": page_num}
                )
                documents.append(doc)
        
        pdf_document.close()
        return documents
        
    except Exception as e:
        print(f"❌ OCR extraction failed: {e}")
        return []

def build_vectorstore(pdf_files, thread_id):
    """Loads multiple PDFs, splits, and builds/updates FAISS vector store for specific thread."""
    all_chunks = []
    filenames = []
    
    for pdf_file in pdf_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Extract text (with OCR fallback)
            docs = extract_text_with_ocr(tmp_path)
            
            # Add filename to metadata
            for doc in docs:
                doc.metadata["filename"] = pdf_file.name
                doc.metadata["thread_id"] = thread_id
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            filenames.append(pdf_file.name)
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    if all_chunks:
        vector_path = get_thread_vector_path(thread_id)
        
        # Load existing vectorstore or create new one
        if os.path.exists(vector_path):
            vectorstore = load_vectorstore(thread_id)
            # Add new documents to existing store
            vectorstore.add_documents(all_chunks)
        else:
            # Create new vectorstore
            vectorstore = FAISS.from_documents(all_chunks, embeddings)
        
        vectorstore.save_local(vector_path)
        
        # Save metadata
        save_thread_metadata(thread_id, filenames)
        
        return vectorstore, len(all_chunks)
    
    return None, 0

def load_vectorstore(thread_id):
    """Load existing FAISS vector store for specific thread."""
    vector_path = get_thread_vector_path(thread_id)
    if os.path.exists(vector_path):
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    return None

def get_uploaded_documents(thread_id):
    """Get list of uploaded document filenames for specific thread."""
    return get_thread_documents(thread_id)

# =====================================================================
# Define LangGraph State
# =====================================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    retrieved_context: str
    thread_id: str

# =====================================================================
# RAG Nodes
# =====================================================================

def retrieve_node(state: ChatState):
    """Retrieve relevant context from documents using current query + conversation context."""
    query = state['query']
    thread_id = state.get('thread_id', 'default')
    messages = state.get('messages', [])
    
    # Enhanced query construction with conversation context
    if len(messages) > 1:
        # Include last few exchanges for better context
        recent_context = []
        for msg in messages[-4:]:  # Last 4 messages (2 exchanges)
            if hasattr(msg, 'content'):
                recent_context.append(msg.content)
        
        # Combine current query with recent context for better retrieval
        enhanced_query = f"Recent conversation: {' '.join(recent_context[-2:])} Current question: {query}"
    else:
        enhanced_query = query
    
    # Load thread-specific vectorstore
    vectorstore = load_vectorstore(thread_id)
    
    if vectorstore:
        # Use enhanced query for better document retrieval
        docs = vectorstore.similarity_search(enhanced_query, k=5)
        
        # Format context with source information
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', '')
            page_info = f" (Page {page + 1})" if page != '' else ""
            
            context_parts.append(f"Source: {source}{page_info}\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "(No documents uploaded yet for this conversation. Please upload PDF documents to get started.)"
    
    return {"retrieved_context": context}

def chat_node(state: ChatState):
    """Generate final answer using LLM + context + full conversation history."""
    messages = state["messages"]
    context = state.get("retrieved_context", "")
    
    # Get the latest user message for the main query
    user_msg = messages[-1].content if messages else ""
    
    if context and "(No documents uploaded yet" not in context:
        system_prompt = f"""You are a helpful assistant that answers questions based on the provided document context and conversation history.

CONTEXT FROM DOCUMENTS:
{context}

INSTRUCTIONS:
- Answer the question using the information from the provided documents
- Consider the previous conversation context when answering
- If the answer isn't in the documents, say so clearly
- When referencing information, mention which document it came from
- Be specific and cite relevant sections when possible
- If multiple documents contain relevant information, synthesize the information appropriately
- Maintain context from previous messages in this conversation

Current Question: {user_msg}"""
    else:
        system_prompt = f"""You are a helpful assistant. No documents have been uploaded yet for this conversation, so I can only provide general assistance based on our conversation history.

To get document-specific answers, please upload PDF documents using the upload feature.

Current Question: {user_msg}"""

    # Pass the full message history INCLUDING the system prompt
    full_messages = [SystemMessage(content=system_prompt)] + messages
    
    try:
        response = llm.invoke(full_messages)
        return {"messages": [response]}
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        # Return error message
        from langchain_core.messages import AIMessage
        error_response = AIMessage(content=f"Sorry, I encountered an error: {str(e)}. Please check your Azure OpenAI configuration.")
        return {"messages": [error_response]}

# =====================================================================
# Graph Setup
# =====================================================================
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("retriever", retrieve_node)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "retriever")
graph.add_edge("retriever", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
