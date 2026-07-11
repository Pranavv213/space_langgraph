import os
import csv
import json
import time
import shutil
from typing import List, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="Multi-Format Protected LangGraph RAG Service")

# =====================================================================
# CORS CONFIGURATION
# =====================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# =====================================================================
# GLOBAL EXCEPTION HANDLER
# =====================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"❌ Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# =====================================================================
# 1. GLOBAL INITS & CONFIGS
# =====================================================================

MY_PINECONE_KEY = ""
PASSWORD_DB_FILE = "index_passwords.json"
GOOGLE_API_KEY = ""

# Ensure password file exists
if not os.path.exists(PASSWORD_DB_FILE):
    with open(PASSWORD_DB_FILE, "w") as f:
        json.dump({}, f)
    print(f"✅ Created empty password file: {PASSWORD_DB_FILE}")

print("Initializing Managed Pinecone Cloud Inference Embeddings...")
hf_embeddings = PineconeEmbeddings(
    model="multilingual-e5-large", 
    pinecone_api_key=MY_PINECONE_KEY
)

print("Connecting to hosted Google Gemini pipeline...")
# Use gemini-pro which is universally available
try:
    local_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        streaming=True,
        convert_system_message_to_human=True,
    )
    print(f"✅ Gemini model initialized successfully with gemini-pro")
except Exception as e:
    print(f"❌ Error initializing Gemini: {e}")
    # Fallback to gemini-1.0-pro
    try:
        local_llm = ChatGoogleGenerativeAI(
            model="gemini-1.0-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            streaming=True,
            convert_system_message_to_human=True,
        )
        print(f"✅ Gemini model initialized successfully with gemini-1.0-pro")
    except Exception as e2:
        print(f"❌ All Gemini models failed: {e2}")
        raise Exception(f"Failed to initialize any Gemini model: {e2}")

pc = Pinecone(api_key=MY_PINECONE_KEY)

# =====================================================================
# 2. PERSISTENCE HELPERS & SECURITY LAYER
# =====================================================================

def load_passwords() -> dict:
    """Loads saved passwords from a persistent local JSON register."""
    if os.path.exists(PASSWORD_DB_FILE):
        try:
            with open(PASSWORD_DB_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing passwords file: {e}")
            return {}
        except Exception as e:
            print(f"⚠️ Error loading passwords: {e}")
            return {}
    return {}

def save_password(index_name: str, password: str):
    """Saves a password configuration permanently to disk."""
    passwords = load_passwords()
    passwords[index_name] = password
    with open(PASSWORD_DB_FILE, "w") as f:
        json.dump(passwords, f, indent=4)
    print(f"✅ Password saved for index: {index_name}")

def verify_index_access(index_name: str, password_provided: str):
    """
    Validates requests. If an index registry entry exists, 
    the provided password must match it perfectly.
    """
    passwords = load_passwords()
    print(f"🔍 Verifying access for index: '{index_name}'")
    print(f"📋 Passwords loaded: {passwords}")
    
    if index_name in passwords:
        stored_password = passwords[index_name]
        print(f"🔑 Stored password: '{stored_password}'")
        print(f"🔑 Provided password: '{password_provided}'")
        
        if stored_password != password_provided:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid password for this index.")
        else:
            print(f"✅ Password matched for index: '{index_name}'")
            return True
    else:
        print(f"⚠️ No password configured for index '{index_name}'. Saving provided password.")
        save_password(index_name, password_provided)
        return True

# =====================================================================
# 3. REQUEST SCHEMAS & UTILITIES
# =====================================================================

class IndexCreationRequest(BaseModel):
    index_name: str
    password: str
    cloud: Optional[str] = "aws"
    region: Optional[str] = "us-east-1"

class QueryRequest(BaseModel):
    index_name: str
    password: str
    question: str

def get_vector_store(index_name: str) -> PineconeVectorStore:
    try:
        existing_indexes = [idx['name'] for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' does not exist on Pinecone.")
        
        return PineconeVectorStore(
            index_name=index_name,
            embedding=hf_embeddings,
            pinecone_api_key=MY_PINECONE_KEY
        )
    except Exception as e:
        print(f"❌ Error getting vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Error connecting to Pinecone: {str(e)}")

def extract_text_by_filetype(file_path: str, filename: str) -> str:
    ext = os.path.splitext(filename)[-1].lower()
    full_text = ""

    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()

    elif ext == ".csv":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            current_batch = []
            ROW_BATCH_SIZE = 50 
            
            for idx, row in enumerate(reader):
                row_items = [f"{key.strip()}: {value.strip()}" for key, value in row.items() if key and value]
                current_batch.append(f"Record line {idx + 1}: {', '.join(row_items)}.")
                
                if len(current_batch) >= ROW_BATCH_SIZE:
                    full_text += "\n".join(current_batch) + "\n\n"
                    current_batch = [] 
            
            if current_batch:
                full_text += "\n".join(current_batch) + "\n\n"
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format '{ext}'. Only .pdf, .txt, and .csv are permitted."
        )

    return full_text

# =====================================================================
# 4. PROTECTED API ENDPOINTS
# =====================================================================

@app.post("/create")
async def create_index(payload: IndexCreationRequest):
    """Creates a fresh 1024-dimension Index and pairs it with a unique security password."""
    try:
        if len(payload.password.strip()) < 4:
            raise HTTPException(status_code=400, detail="Password must be at least 4 characters long.")
        
        existing_indexes = [idx['name'] for idx in pc.list_indexes()]
        
        if payload.index_name in existing_indexes:
            save_password(payload.index_name, payload.password)
            return {
                "status": "exists", 
                "message": f"Index '{payload.index_name}' already exists. Password recorded."
            }
        
        print(f"Provisioning fresh index '{payload.index_name}'...")
        pc.create_index(
            name=payload.index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud=payload.cloud, region=payload.region)
        )
        
        time.sleep(3)
        save_password(payload.index_name, payload.password)
        
        return {
            "status": "success", 
            "message": f"Index '{payload.index_name}' provisioned and secured."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add")
async def add_document(
    index_name: str = Form(...), 
    password: str = Form(...),
    file: UploadFile = File(...)
):
    """Parses and indexes text contents safely if authorization credentials pass."""
    try:
        verify_index_access(index_name, password)
        vector_store = get_vector_store(index_name)
        
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            full_text = extract_text_by_filetype(temp_path, file.filename)
            if not full_text.strip():
                raise HTTPException(status_code=400, detail="The data source yielded no indexable text contents.")
                
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            text_chunks = text_splitter.split_text(full_text)
            
            metadatas = [{"source_file": file.filename} for _ in text_chunks]
            
            RATE_LIMIT_BATCH_SIZE = 10
            DELAY_BETWEEN_BATCHES = 3.0
            
            for i in range(0, len(text_chunks), RATE_LIMIT_BATCH_SIZE):
                batch_texts = text_chunks[i : i + RATE_LIMIT_BATCH_SIZE]
                batch_metadatas = metadatas[i : i + RATE_LIMIT_BATCH_SIZE]
                
                vector_store.add_texts(
                    texts=batch_texts, 
                    metadatas=batch_metadatas, 
                    batch_size=RATE_LIMIT_BATCH_SIZE
                )
                
                if i + RATE_LIMIT_BATCH_SIZE < len(text_chunks):
                    time.sleep(DELAY_BETWEEN_BATCHES)

            return {
                "status": "success", 
                "chunks_uploaded": len(text_chunks), 
                "file": file.filename
            }
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
async def update_document(
    index_name: str = Form(...), 
    password: str = Form(...),
    old_file_name: str = Form(...), 
    new_file: UploadFile = File(...)
):
    """Purges chunks matching an old filename, then parses and replaces it."""
    try:
        verify_index_access(index_name, password)
        vector_store = get_vector_store(index_name)
        
        print(f"Purging old document '{old_file_name}' from index '{index_name}'...")
        try:
            vector_store.delete(filter={"source_file": {"$eq": old_file_name}})
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
            
            RATE_LIMIT_BATCH_SIZE = 10
            DELAY_BETWEEN_BATCHES = 3.0
            
            for i in range(0, len(text_chunks), RATE_LIMIT_BATCH_SIZE):
                batch_texts = text_chunks[i : i + RATE_LIMIT_BATCH_SIZE]
                batch_metadatas = metadatas[i : i + RATE_LIMIT_BATCH_SIZE]
                vector_store.add_texts(
                    texts=batch_texts, 
                    metadatas=batch_metadatas, 
                    batch_size=RATE_LIMIT_BATCH_SIZE
                )
                if i + RATE_LIMIT_BATCH_SIZE < len(text_chunks):
                    time.sleep(DELAY_BETWEEN_BATCHES)
            
            return {
                "status": "success", 
                "purged_file": old_file_name,
                "inserted_file": new_file.filename,
                "chunks_uploaded": len(text_chunks)
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# 5. QUERY ENDPOINT WITH PROPER STREAMING
# =====================================================================

@app.post("/query")
async def query_rag(payload: QueryRequest):
    """Streams responses using direct retrieval and generation."""
    try:
        verify_index_access(payload.index_name, payload.password)
        vector_store = get_vector_store(payload.index_name)
        
        async def generate_response():
            try:
                # Step 1: Retrieve relevant documents
                print(f"🔍 Retrieving documents for query: {payload.question[:50]}...")
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(payload.question)
                
                if not retrieved_docs:
                    yield "I don't have any documents in this workspace to answer your question. Please upload some documents first."
                    return
                
                # Step 2: Prepare context
                context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                print(f"📄 Retrieved {len(retrieved_docs)} documents with {len(context)} characters")
                
                # Step 3: Create prompt
                system_prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

Context:
{context}

Instructions:
- Answer the user's question using only the information from the context above
- If the context doesn't contain the answer, say "I don't have information about that in the documents"
- Be concise and direct
- Do not make up information"""

                # Step 4: Generate response with streaming
                print("🤖 Generating response with Gemini...")
                response = local_llm.stream([
                    ("system", system_prompt),
                    ("human", payload.question)
                ])
                
                # Step 5: Stream the response - properly handling different chunk types
                for chunk in response:
                    if chunk.content:
                        # Handle different content types
                        if isinstance(chunk.content, str):
                            yield chunk.content
                        elif isinstance(chunk.content, list):
                            # If content is a list, extract text from each item
                            for item in chunk.content:
                                if isinstance(item, str):
                                    yield item
                                elif isinstance(item, dict) and 'text' in item:
                                    yield item['text']
                                else:
                                    yield str(item)
                        else:
                            # Fallback: convert to string
                            yield str(chunk.content)
                
                print("✅ Response streaming complete")
                
            except Exception as e:
                print(f"❌ Error in generate_response: {str(e)}")
                yield f"Error: {str(e)}"
        
        return StreamingResponse(
            generate_response(), 
            media_type="text/plain"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# 6. DEBUG ENDPOINTS
# =====================================================================

@app.get("/debug/passwords")
async def debug_passwords():
    """Debug endpoint to check stored passwords."""
    try:
        passwords = load_passwords()
        return {
            "passwords_file_exists": os.path.exists(PASSWORD_DB_FILE),
            "passwords": passwords,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/indexes")
async def debug_indexes():
    """Debug endpoint to list all Pinecone indexes."""
    try:
        indexes = pc.list_indexes()
        return {
            "indexes": [idx['name'] for idx in indexes],
            "total": len(indexes)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/test-gemini")
async def test_gemini():
    """Test Gemini connection with the current model."""
    try:
        response = local_llm.invoke([
            ("system", "You are a helpful assistant. Respond with a short greeting."),
            ("human", "Say hello and confirm you are working")
        ])
        return {
            "status": "success",
            "model": "gemini-pro",
            "response": response.content,
            "model_type": str(type(local_llm))
        }
    except Exception as e:
        return {
            "status": "error",
            "model": "gemini-pro",
            "error": str(e),
            "error_type": str(type(e))
        }

@app.get("/debug/test-retrieval/{index_name}")
async def test_retrieval(index_name: str, question: str = "test"):
    """Test retrieval from Pinecone."""
    try:
        vector_store = get_vector_store(index_name)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(question)
        return {
            "index_name": index_name,
            "question": question,
            "num_docs": len(docs),
            "documents": [{"content": doc.page_content[:200], "metadata": doc.metadata} for doc in docs]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "gemini-pro",
        "pinecone": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)