import os
import json
import asyncio
import fitz
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.schema import Document

SUMMARY_CACHE_FILE = 'patient_summary_cache.json'
DOCUMENT_MANIFEST_FILE = 'document_manifest.json'

def read_pdf_file_robust(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"    - ❗️ Error reading {os.path.basename(file_path)} with PyMuPDF: {e}")
        return ""
    return text

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_all_patient_indexes(data_root='data'):
    print("Starting real index loading process for all patients...")
    patient_indexes = {}
    llm = OpenAI(model="gpt-4", temperature=0)
    Settings.llm = llm

    for patient_name_raw in os.listdir(data_root):
        patient_name = patient_name_raw.strip()  # Clean whitespace
        patient_folder = os.path.join(data_root, patient_name_raw)
        if not os.path.isdir(patient_folder):
            continue
        print(f"\nProcessing patient: {patient_name}")
        documents = []
        for filename in os.listdir(patient_folder):
            file_path = os.path.join(patient_folder, filename)
            text = ""
            if filename.lower().endswith('.pdf'):
                text = read_pdf_file_robust(file_path)
            elif filename.lower().endswith('.txt'):
                text = read_text_file(file_path)
            if text.strip():
                documents.append(Document(text=text, doc_id=file_path))
        if not documents:
            print(f"  - No readable documents found for {patient_name}, skipping index creation.")
            continue
        print(f"  - Creating AI index from {len(documents)} documents...")
        index = VectorStoreIndex.from_documents(documents)
        patient_indexes[patient_name] = index
        print(f"  - ✅ Index created for {patient_name}.")
    return patient_indexes

async def classify_document(filepath: str, llm: OpenAI):
    print(f"  - Classifying '{os.path.basename(filepath)}'...")
    content_snippet = ""
    if filepath.lower().endswith('.pdf'):
        content_snippet = read_pdf_file_robust(filepath).strip()[:2000]
    else:
        content_snippet = read_text_file(filepath).strip()[:2000]
    if not content_snippet:
        print("    - Could not read content, skipping classification.")
        return "Other"
    prompt = f"""Based on the following text from a medical document, classify it into ONE of the following categories: [Clinical Note, Lab Result, Prescription, Imaging Report, Insurance, Other]. Respond with ONLY the category name and nothing else. Text snippet: --- {content_snippet} --- Category:"""
    response = await llm.acomplete(prompt)
    category = response.text.strip()
    valid_categories = ["Clinical Note", "Lab Result", "Prescription", "Imaging Report", "Insurance", "Other"]
    if category not in valid_categories:
        return "Other"
    print(f"    - Classified as: {category}")
    return category

async def generate_summary_for_patient(patient_name: str, index: VectorStoreIndex):
    print(f"  - Generating AI summary for {patient_name}...")
    try:
        query_engine = index.as_query_engine(llm=Settings.llm)
        queries = {
            "medication_summary": "Provide a clear summary of the patient's prescribed medications...",
            "lifestyle_recommendations": "Based on the patient's medical records, list key lifestyle recommendations.",
            "condition_summary": "Summarize the patient's diagnosed conditions..."
        }
        tasks = [query_engine.aquery(q) for q in queries.values()]
        responses = await asyncio.gather(*tasks)
        summary_data = {
            "medication_summary": str(responses[0]),
            "lifestyle_recommendations": str(responses[1]),
            "condition_summary": str(responses[2]),
        }
        print(f"  - ✅ Summary generated for {patient_name}.")
        return patient_name, summary_data
    except Exception as e:
        print(f"  - ❌ FAILED to generate summary for {patient_name}: {e}")
        return patient_name, None

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
        return
    llm = OpenAI(model="gpt-4", temperature=0)
    Settings.llm = llm
    print("--- Starting Part 1: Generating Patient Summaries ---")
    patient_indexes = load_all_patient_indexes()
    if os.path.exists(SUMMARY_CACHE_FILE):
        with open(SUMMARY_CACHE_FILE, 'r') as f:
            summary_cache = json.load(f)
    else:
        summary_cache = {}

    tasks = []
    for name, index in patient_indexes.items():
        if name not in summary_cache:
            tasks.append(generate_summary_for_patient(name, index))

    if tasks:
        results = await asyncio.gather(*tasks)
        for name, summary_data in results:
            if summary_data:
                summary_cache[name] = summary_data

    with open(SUMMARY_CACHE_FILE, 'w') as f:
        json.dump(summary_cache, f, indent=2)

    print(f"✅ Part 1 complete. Summaries saved to {SUMMARY_CACHE_FILE}.")

    print("\n--- Starting Part 2: Classifying Patient Documents ---")
    if os.path.exists(DOCUMENT_MANIFEST_FILE):
        with open(DOCUMENT_MANIFEST_FILE, 'r') as f:
            document_manifest = json.load(f)
    else:
        document_manifest = {}

    all_patients = os.listdir('data')

    for patient_name_raw in all_patients:
        patient_name = patient_name_raw.strip()
        patient_folder = os.path.join('data', patient_name_raw)
        if not os.path.isdir(patient_folder):
            continue

        if patient_name not in document_manifest:
            document_manifest[patient_name] = []

        print(f"\nProcessing documents for: {patient_name}")

        processed_files = [doc['filename'] for doc in document_manifest[patient_name]]
        classification_tasks = []
        files_to_process = []

        for filename in os.listdir(patient_folder):
            if filename in processed_files:
                continue
            filepath = os.path.join(patient_folder, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(('.pdf', '.txt')):
                classification_tasks.append(classify_document(filepath, llm))
                files_to_process.append({'filename': filename, 'path': filepath})

        if classification_tasks:
            categories = await asyncio.gather(*classification_tasks)
            for i, category in enumerate(categories):
                doc_info = files_to_process[i]
                document_manifest[patient_name].append({
                    "filename": doc_info['filename'],
                    "category": category,
                    "path": doc_info['path']
                })

    with open(DOCUMENT_MANIFEST_FILE, 'w') as f:
        json.dump(document_manifest, f, indent=2)

    print(f"\n✅ Processing complete. Document manifest saved to {DOCUMENT_MANIFEST_FILE}.")

if __name__ == "__main__":
    asyncio.run(main())

