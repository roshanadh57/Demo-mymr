import os
import json
import threading
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, List
from contextlib import asynccontextmanager
import traceback
import fitz  # PyMuPDF
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from llama_index.core import Settings, PromptTemplate
from llama_index.core.schema import Document
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Settings.llm = None

# ------------------ Google GenAI SDK ------------------
from google import genai

# Initialize GenAI client once
client = genai.Client(api_key="AIzaSyC5qpkHcyRoFtaiNdcFbXnZBbX7fNdlB9c")

# The Gemini model you want to use for generation
MODEL_NAME = "gemini-2.5-flash"  # Replace with your desired model

def query_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text

# ------------------ Embeddings ------------------
print("🔧 Setting up embeddings...")
hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(model_name=hf_model_name)
Settings.embed_model = embed_model

# ------------------ Custom QA prompt template ------------------
QA_TEMPLATE = PromptTemplate(
    """\
You are a knowledgeable medical AI assistant. Your task is to answer questions about patients based on their medical records and documents.
CONTEXT INFORMATION:
{context_str}
INSTRUCTIONS:
- Provide detailed, accurate answers based only on the medical records provided
- If asked about medications, list ALL medications with dosages and frequencies
- If asked about medical conditions, provide a comprehensive summary
- If asked about recommendations, explain the medical rationale
- Use clear, professional medical language
- If specific information is not in the records, state this clearly
- Do not make up or assume information not present in the documents
PATIENT QUESTION: {query_str}
DETAILED ANSWER:"""
)

# ------------------ Globals ------------------
patient_indexes: Dict[str, VectorStoreIndex] = {}
patient_summaries: Dict[str, Dict] = {}
document_manifest: Dict[str, List] = {}
last_summary_mtime = 0

# ------------------ Utilities ------------------
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf_file_robust(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""
    return text

def load_documents_from_directory_recursive(root_dir):
    documents = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            ext = file.lower().split('.')[-1]
            text = ""
            if ext == 'txt':
                text = read_text_file(file_path)
            elif ext == 'pdf':
                text = read_pdf_file_robust(file_path)
            else:
                continue
            if text.strip():
                documents.append(Document(text=text, doc_id=file_path))
    return documents

def create_index_for_patient(patient_folder):
    documents = load_documents_from_directory_recursive(patient_folder)
    if not documents:
        return None
    return VectorStoreIndex.from_documents(documents)

def load_all_patient_indexes(data_root='data'):
    indexes = {}
    if not os.path.isdir(data_root):
        return indexes
    for patient_name_raw in os.listdir(data_root):
        patient_name = patient_name_raw.strip()
        patient_folder = os.path.join(data_root, patient_name_raw)
        if os.path.isdir(patient_folder):
            print(f"📁 Loading index for patient: {patient_name}")
            index = create_index_for_patient(patient_folder)
            if index:
                indexes[patient_name] = index
                print(f"✅ Index loaded for {patient_name}")
            else:
                print(f"⚠️ No documents found for {patient_name}")
    return indexes

# ------------------ Watchdog for auto-update ------------------
class DataFolderWatcher(FileSystemEventHandler):
    def __init__(self):
        self._debounce = False

    def on_any_event(self, event):
        ignored_files = [
            'patient_summary_cache.json',
            'document_manifest.json',
            '.tmp',
            '.swp',
            '~',
            '__pycache__'
        ]
        if any(ignored in event.src_path for ignored in ignored_files):
            return
        if (event.event_type in ('created', 'modified') and not event.is_directory) or \
           (event.event_type == 'deleted' and event.is_directory):
            self.trigger_regeneration()

    def trigger_regeneration(self):
        global last_summary_mtime, patient_summaries, document_manifest, patient_indexes
        try:
            current_mtime = os.path.getmtime('patient_summary_cache.json')
        except FileNotFoundError:
            current_mtime = 0
        if current_mtime == last_summary_mtime:
            return
        if not self._debounce:
            self._debounce = True
            threading.Timer(5, self._reset_debounce).start()
            print("🔄 Change detected in data folder, regenerating summaries and indexes...")
            try:
                subprocess.run(["python3", "generate_summaries.py"], check=True)
                with open('patient_summary_cache.json', 'r') as f:
                    patient_summaries = json.load(f)
                with open('document_manifest.json', 'r') as f:
                    document_manifest = json.load(f)
                last_summary_mtime = os.path.getmtime('patient_summary_cache.json')
                patient_indexes = load_all_patient_indexes()
                print("✅ Summaries and patient indexes reloaded successfully.")
            except Exception as e:
                print(f"❌ Error during regeneration: {e}")
                traceback.print_exc()

    def _reset_debounce(self):
        self._debounce = False

# ------------------ FastAPI app ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global patient_indexes, patient_summaries, document_manifest
    print("🏥 Loading patient vector indexes for chat...")
    patient_indexes = load_all_patient_indexes()

    try:
        with open('patient_summary_cache.json', 'r') as f:
            patient_summaries = json.load(f)
        print("✅ Patient summaries loaded successfully.")
    except FileNotFoundError:
        print("⚠️ patient_summary_cache.json not found. Please run generate_summaries.py")
        patient_summaries = {}

    try:
        with open('document_manifest.json', 'r') as f:
            document_manifest = json.load(f)
        print("✅ Document manifest loaded successfully.")
    except FileNotFoundError:
        print("⚠️ document_manifest.json not found. Please run generate_summaries.py")
        document_manifest = {}

    observer = Observer()
    event_handler = DataFolderWatcher()
    observer.schedule(event_handler, path='data', recursive=True)
    observer.start()
    print("👀 Started watchdog observer monitoring 'data/' folder.")

    try:
        yield
    finally:
        observer.stop()
        observer.join()

app = FastAPI(lifespan=lifespan)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Request models ------------------
class QueryRequest(BaseModel):
    patient_name: str
    query: str

class DocumentContentRequest(BaseModel):
    path: str

# ------------------ Routes ------------------
@app.get("/patients")
def get_patients():
    return {"patients": list(patient_summaries.keys())}

@app.get("/summary/{patient_name}")
async def get_summary(patient_name: str):
    summary_data = patient_summaries.get(patient_name)
    if not summary_data:
        raise HTTPException(status_code=404, detail="Summary for patient not found.")
    return {
        "summary": {
            "medication_summary": summary_data.get("medication_summary", "No medication information available."),
            "lifestyle_recommendations": summary_data.get("lifestyle_recommendations", "No lifestyle recommendations available."),
            "condition_summary": summary_data.get("condition_summary", "No condition information available.")
        }
    }

@app.get("/documents/{patient_name}")
async def get_patient_documents(patient_name: str):
    documents = document_manifest.get(patient_name, [])
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found for this patient.")
    return documents

@app.post("/document_content")
async def get_document_content(request: DocumentContentRequest):
    file_path = request.path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        content = ""
        if file_path.lower().endswith('.pdf'):
            content = read_pdf_file_robust(file_path)
        elif file_path.lower().endswith('.txt'):
            content = read_text_file(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        if not content.strip():
            raise HTTPException(status_code=400, detail="Could not read document content.")

        classification = "Other"
        for docs in document_manifest.values():
            for doc in docs:
                if doc["path"] == file_path:
                    classification = doc.get("category", "Other")
                    break
        return {"content": content, "classification": classification}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading document: {str(e)}")

@app.post("/query")
async def query_patient(data: QueryRequest):
    index = patient_indexes.get(data.patient_name)
    if not index:
        raise HTTPException(status_code=404, detail="Patient index not found")
    try:
        print(f"🔍 Processing query for {data.patient_name}: {data.query}")

        # create a query engine from the vector index
        query_engine = index.as_query_engine()

        # get response from the query engine
        response = query_engine.query(data.query)

        # convert response to string to send to Gemini or directly return
        retrieved_text = str(response)

        # build prompt with retrieved context + user query
        prompt = QA_TEMPLATE.format(context_str=retrieved_text, query_str=data.query)

        # call Gemini with full prompt
        response_text = query_gemini(prompt)

        print("✅ Response generated successfully.")
        return {"answer": response_text}

    except Exception as e:
        print(f"❌ Error during query processing: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "embedding_model": hf_model_name,
        "patients_loaded": len(patient_indexes),
        "summaries_loaded": len(patient_summaries),
        "documents_loaded": len(document_manifest)
    }

@app.get("/")
def read_root():
    return {
        "message": "🏥 Patient Chat API with Gemini LLM",
        "embedding_model": hf_model_name,
        "patients_loaded": len(patient_indexes),
        "summaries_loaded": len(patient_summaries),
        "documents_loaded": len(document_manifest),
        "status_endpoint": "/status"
    }

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# ------------------ Run server ------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Patient Chat API with Gemini")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
