import os
import time
from pypdf import PdfReader
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph, START, END

# =====================================================================
# 1. SETUP CONNECTIONS & INITS (HUGGING FACE, PINECONE, OLLAMA)
# =====================================================================

MY_PINECONE_KEY = "pcsk_6cPP3h_KD6NTCcu3eZLpYv7deCKR5sVckjqhfwwY8Cz7d8bYFcJ9xFog1wbKHh2BJBFWAZ"
index_name = "simple-rag"

# Initialize Hugging Face embeddings locally
print("Initializing Hugging Face embeddings...")
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5", # Generates 1024-dimension vectors to match your index
    model_kwargs={"device": "cpu"}
)

# Connect to cloud Pinecone Index directly using your API key string
print("Connecting to cloud Pinecone Index...")
vector_store = PineconeVectorStore(
    index_name=index_name, 
    embedding=hf_embeddings,
    pinecone_api_key=MY_PINECONE_KEY
)

# Setup Local Chat Inference engine via Ollama 
# (Tip: Use "llama3.2:3b" for faster execution speeds on standard laptop hardware)
print("Connecting to local Ollama instance...")
local_llm = ChatOllama(model="llama3.2:3b", temperature=0)


# =====================================================================
# 2. DATA MANAGEMENT FUNCTIONS (UPLOAD, DELETE, UPDATE)
# =====================================================================

def add_pdf_to_db(file_path: str):
    """
    Accepts a local PDF path string, extracts text, chunks it,
    tags it with the file name as metadata, and pushes it to Pinecone.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file path '{file_path}' does not exist.")
        return

    filename = os.path.basename(file_path)
    print(f"Reading PDF contents from: {file_path}")
    reader = PdfReader(file_path)
    full_text = ""
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    if not full_text.strip():
        print(f"Warning: Could not extract any readable text from {filename}.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    text_chunks = text_splitter.split_text(full_text)
    
    # Track the origin file name inside metadata for targeted updates/deletions later
    metadatas = [{"source_file": filename} for _ in text_chunks]
    
    print(f"Generating embeddings and uploading {len(text_chunks)} chunks for '{filename}'...")
    vector_store.add_texts(texts=text_chunks, metadatas=metadatas, batch_size=128)
    print(f"Successfully indexed '{filename}' into database.")


def replace_document_in_db(old_file_name: str, new_file_path: str):
    """
    Purges all vector chunks linked to an old filename metadata value,
    then processes and uploads a new document to replace it.
    """
    print(f"\n--- Starting Replacement Process ---")
    print(f"1. Locating and purging all vectors associated with: '{old_file_name}'...")
    
    try:
        # Access underlying pinecone index instance via the LangChain wrapper
        vector_store.index.delete(
            filter={"source_file": {"$eq": old_file_name}}
        )
        print("   Purge successful.")
    except Exception as e:
        print(f"   Note/Error during deletion phase: {e}")

    print(f"2. Uploading replacement file: '{new_file_path}'...")
    add_pdf_to_db(new_file_path)
    print(f"--- Replacement Process Complete ---\n")


# =====================================================================
# 3. LANGGRAPH WORKFLOW CORE
# =====================================================================

def simple_rag_node(state: MessagesState):
    """Retrieves context from Pinecone and generates an answer using Ollama."""
    messages = state["messages"]
    last_user_query = messages[-1].content
    
    # Pull the 2 closest matching blocks from Pinecone
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(last_user_query)
    
    # Combine the retrieved text content
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Inject context into systemic prompt guidelines
    system_prompt = (
        f"You are a helpful assistant.\n"
        f"Answer the user query completely using only the text blocks provided below.\n\n"
        f"Context Blocks:\n{context}"
    )
    
    # Compile messages and invoke inference
    compiled_messages = [("system", system_prompt)] + messages
    model_response = local_llm.invoke(compiled_messages)
    
    return {"messages": [model_response]}

# Compile the single-node state graph
workflow = StateGraph(MessagesState)
workflow.add_node("rag_core", simple_rag_node)
workflow.add_edge(START, "rag_core")
workflow.add_edge("rag_core", END)

app = workflow.compile()


# =====================================================================
# 4. EXECUTION
# =====================================================================

if __name__ == "__main__":
    
    # -----------------------------------------------------------------
    # USE CASE A: FRESH UPLOAD OR SYSTEM SETUP
    # -----------------------------------------------------------------
    pdf_path = "myth.pdf"
    # add_pdf_to_db(file_path=pdf_path) # <-- Run once to populate, then leave commented out
    
    # -----------------------------------------------------------------
    # USE CASE B: REPLACING AN OLD FILE WITH A NEW ONE
    # -----------------------------------------------------------------
    # If you want to overwrite a document you uploaded previously, uncomment this:
    # replace_document_in_db(old_file_name="astrology.pdf", new_file_path="myth.pdf")

    # -----------------------------------------------------------------
    # RUNNING INFERENCE QUESTIONS (STREAMING OUTPUT)
    # -----------------------------------------------------------------
    query_inputs = {"messages": [("user", "who was narada ?")]}
    
    print("\nRunning RAG Query via LangGraph...")
    print("\n--- Ollama Response ---")
    
    # stream_mode="messages" streams individual tokens instantly out of the model
    for chunk, metadata in app.stream(query_inputs, stream_mode="messages"):
        if chunk.content and metadata.get("langgraph_node") == "rag_core":
            print(chunk.content, end="", flush=True)
            
    print("\n\n--- Stream Finished ---")