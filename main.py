import os
import csv
import time
import shutil
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_pinecone import PineconeEmbeddings

app = FastAPI(title="Multi-Format Dynamic LangGraph RAG Service")

# =====================================================================
# 1. GLOBAL INITS & CONFIGS
# =====================================================================

MY_PINECONE_KEY = "pcsk_6cPP3h_KD6NTCcu3eZLpYv7deCKR5sVckjqhfwwY8Cz7d8bYFcJ9xFog1wbKHh2BJBFWAZ"

print("Initializing Managed Pinecone Cloud Inference Embeddings...")
hf_embeddings = PineconeEmbeddings(
    model="multilingual-e5-large", # Natively outputs 1024 dimensions
    pinecone_api_key=MY_PINECONE_KEY
)

print("Connecting to hosted Google Gemini pipeline...")
# Replaces your old local_llm or ChatOllama block
local_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="AIzaSyBp_c9NdPOKypYBsniEKz_q0Ahyi4Wljlc",  # <--- Pass your key directly here
    temperature=0,
    streaming=True
)

# Instantiate raw client for dashboard configuration tasks
pc = Pinecone(api_key=MY_PINECONE_KEY)


# =====================================================================
# 2. SCHEMAS & HELPERS
# =====================================================================

class IndexCreationRequest(BaseModel):
    index_name: str
    cloud: Optional[str] = "aws"
    region: Optional[str] = "us-east-1"

class QueryRequest(BaseModel):
    index_name: str
    question: str


def get_vector_store(index_name: str) -> PineconeVectorStore:
    """Helper to cleanly connect the LangChain wrapper to a target index."""
    existing_indexes = [idx['name'] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' does not exist on Pinecone.")
    
    return PineconeVectorStore(
        index_name=index_name,
        embedding=hf_embeddings,
        pinecone_api_key=MY_PINECONE_KEY
    )


def extract_text_by_filetype(file_path: str, filename: str) -> str:
    """
    Inspects the file extension and extracts readable text.
    Batches CSV rows together to avoid massive CPU local embedding bottlenecks.
    """
    ext = os.path.splitext(filename)[-1].lower()
    full_text = ""

    # --- HANDLE PDF FILES ---
    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # --- HANDLE TEXT FILES ---
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()

    # --- HANDLE CSV FILES (50-Row Batching Engine) ---
    elif ext == ".csv":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            
            current_batch = []
            ROW_BATCH_SIZE = 50 
            
            for idx, row in enumerate(reader):
                # Format columns into single semantic row string
                row_items = [f"{key.strip()}: {value.strip()}" for key, value in row.items() if key and value]
                current_batch.append(f"Record line {idx + 1}: {', '.join(row_items)}.")
                
                # Once we hit 50 rows, merge them and flush to full_text
                if len(current_batch) >= ROW_BATCH_SIZE:
                    full_text += "\n".join(current_batch) + "\n\n"
                    current_batch = [] 
            
            # Flush any remaining lines left over at the end of the file
            if current_batch:
                full_text += "\n".join(current_batch) + "\n\n"

    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format '{ext}'. Only .pdf, .txt, and .csv are permitted."
        )

    return full_text


# =====================================================================
# 3. FASTAPI API ENDPOINTS
# =====================================================================

@app.post("/create")
def create_index(payload: IndexCreationRequest):
    """Creates a fresh 1024-dimension Serverless Pinecone Index."""
    existing_indexes = [idx['name'] for idx in pc.list_indexes()]
    if payload.index_name in existing_indexes:
        return {"message": f"Index '{payload.index_name}' already exists."}
    
    print(f"Provisioning fresh index '{payload.index_name}'...")
    try:
        pc.create_index(
            name=payload.index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud=payload.cloud, region=payload.region)
        )
        return {"status": "success", "message": f"Index '{payload.index_name}' provisioned successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add")
async def add_document(index_name: str = Form(...), file: UploadFile = File(...)):
    """Parses an uploaded file (.pdf/.txt/.csv), tags metadata, and saves chunks to Pinecone."""
    vector_store = get_vector_store(index_name)
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        full_text = extract_text_by_filetype(temp_path, file.filename)
            
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="The data source yielded no indexable text contents.")
            
        # Recursive Character splitter groups text efficiently based on token windows
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        text_chunks = text_splitter.split_text(full_text)
        
        metadatas = [{"source_file": file.filename} for _ in text_chunks]
        
        # batch_size=128 optimizes the matrix arrays sent down to the embedding model
        vector_store.add_texts(texts=text_chunks, metadatas=metadatas, batch_size=128)
        return {"status": "success", "chunks_uploaded": len(text_chunks), "file": file.filename}
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/update")
async def update_document(
    index_name: str = Form(...), 
    old_file_name: str = Form(...), 
    new_file: UploadFile = File(...)
):
    """Purges chunks matching an old filename string, then parses and uploads the new file in its place."""
    vector_store = get_vector_store(index_name)
    
    print(f"Purging old document '{old_file_name}' from index '{index_name}'...")
    try:
        vector_store.index.delete(filter={"source_file": {"$eq": old_file_name}})
    except Exception as e:
        print(f"Note during purge phase: {e}")
        
    temp_path = f"temp_{new_file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(new_file.file, buffer)
        
    try:
        full_text = extract_text_by_filetype(temp_path, new_file.filename)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        text_chunks = text_splitter.split_text(full_text)
        
        metadatas = [{"source_file": new_file.filename} for _ in text_chunks]
        vector_store.add_texts(texts=text_chunks, metadatas=metadatas, batch_size=128)
        
        return {
            "status": "success", 
            "purged_file": old_file_name,
            "inserted_file": new_file.filename,
            "chunks_uploaded": len(text_chunks)
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================================
# 4. LANGGRAPH WORKFLOW CORE & STREAM ROUTER
# =====================================================================

def make_rag_node(vector_store: PineconeVectorStore):
    """Dynamically generates a graph node closed over a specific index target store."""
    def simple_rag_node(state: MessagesState):
        messages = state["messages"]
        last_user_query = messages[-1].content
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        retrieved_docs = retriever.invoke(last_user_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        system_prompt = (
            f"You are a helpful assistant.\n"
            f"Answer the user query completely using only the text blocks provided below.\n\n"
            f"Context Blocks:\n{context}"
        )
        
        compiled_messages = [("system", system_prompt)] + messages
        model_response = local_llm.invoke(compiled_messages)
        return {"messages": [model_response]}
    return simple_rag_node


@app.post("/query")
async def query_rag(payload: QueryRequest):
    """Streams responses out of the LangGraph runtime engine in real-time."""
    vector_store = get_vector_store(payload.index_name)
    
    workflow = StateGraph(MessagesState)
    workflow.add_node("rag_core", make_rag_node(vector_store))
    workflow.add_edge(START, "rag_core")
    workflow.add_edge("rag_core", END)
    graph_app = workflow.compile()
    
    query_inputs = {"messages": [("user", payload.question)]}
    
    async def token_generator():
        for chunk, metadata in graph_app.stream(query_inputs, stream_mode="messages"):
            if chunk.content and metadata.get("langgraph_node") == "rag_core":
                yield chunk.content

    return StreamingResponse(token_generator(), media_type="text/plain")


# =====================================================================
# 5. SERVER LAUNCH
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
