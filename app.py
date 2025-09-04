import os
import json
import threading
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from contextlib import asynccontextmanager
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.schema import Document
import fitz  # PyMuPDF for robust PDF reading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Global state containers
patient_indexes: Dict[str, VectorStoreIndex] = {}
patient_summaries: Dict[str, Dict] = {}
document_manifest: Dict[str, List] = {}
last_summary_mtime = 0

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
    llm = OpenAI(model="gpt-4", temperature=0)
    Settings.llm = llm
    return VectorStoreIndex.from_documents(documents)

def load_all_patient_indexes(data_root='data'):
    indexes = {}
    for patient_name_raw in os.listdir(data_root):
        patient_name = patient_name_raw.strip()
        patient_folder = os.path.join(data_root, patient_name_raw)
        if os.path.isdir(patient_folder):
            index = create_index_for_patient(patient_folder)
            if index:
                indexes[patient_name] = index
    return indexes

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
        # Ignore events on files we don't care to watch
        if any(ignored in event.src_path for ignored in ignored_files):
            return

        # Trigger regeneration on file created/modified or patient folder deleted (directory deleted)
        if (event.event_type in ('created', 'modified') and not event.is_directory) or \
           (event.event_type == 'deleted' and event.is_directory):
            self.trigger_regeneration()

    def trigger_regeneration(self):
        global last_summary_mtime, patient_summaries, document_manifest, patient_indexes
        try:
            current_mtime = os.path.getmtime('patient_summary_cache.json')
        except FileNotFoundError:
            current_mtime = 0

        # Skip if file modification time hasn't changed to avoid recursion
        if current_mtime == last_summary_mtime:
            return

        if not self._debounce:
            self._debounce = True
            threading.Timer(5, self._reset_debounce).start()
            print("Change detected in data folder, regenerating summaries and indexes...")

            try:
                subprocess.run(["python3", "generate_summaries.py"], check=True)

                # Reload summaries and document manifest from updated cache files
                with open('patient_summary_cache.json', 'r') as f:
                    patient_summaries = json.load(f)
                with open('document_manifest.json', 'r') as f:
                    document_manifest = json.load(f)
                last_summary_mtime = os.path.getmtime('patient_summary_cache.json')

                # Reload patient indexes from current data folder state to reflect deletions/additions
                patient_indexes = load_all_patient_indexes()

                print("Summaries and patient indexes reloaded successfully.")

            except Exception as e:
                print(f"Error during regeneration: {e}")

    def _reset_debounce(self):
        self._debounce = False



@asynccontextmanager
async def lifespan(app: FastAPI):
    global patient_indexes, patient_summaries, document_manifest
    print("Loading patient vector indexes for chat...")
    patient_indexes = load_all_patient_indexes()
    try:
        with open('patient_summary_cache.json', 'r') as f:
            patient_summaries = json.load(f)
        print("✅ Summaries loaded successfully.")
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
    print("Started watchdog observer monitoring 'data/' folder.")
    
    try:
        yield
    finally:
        observer.stop()
        observer.join()


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:3000"]  
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    patient_name: str
    query: str

class DocumentContentRequest(BaseModel):
    path: str

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
def query_patient(data: QueryRequest):
    index = patient_indexes.get(data.patient_name)
    if not index:
        raise HTTPException(status_code=404, detail="Patient index not found")
    if not Settings.llm:
        Settings.llm = OpenAI(model="gpt-4", temperature=0)
    query_engine = index.as_query_engine(llm=Settings.llm)
    response = query_engine.query(data.query)
    return {"answer": str(response)}

@app.get("/")
def read_root():
    return {
        "message": "Patient Chat API is running",
        "patients_loaded": len(patient_indexes),
        "summaries_loaded": len(patient_summaries),
        "documents_loaded": len(document_manifest)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

