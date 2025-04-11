from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Debug output to verify env loading
print("SUPABASE_URL:", os.getenv("SUPABASE_URL"))
print("SUPABASE_SERVICE_KEY:", "set" if os.getenv("SUPABASE_SERVICE_KEY") else "missing")
print("OPENAI_API_KEY:", "set" if os.getenv("OPENAI_API_KEY") else "missing")
print("DOCUMENT_PROCESSING_API_KEY:", "set" if os.getenv("DOCUMENT_PROCESSING_API_KEY") else "missing")

from fastapi import FastAPI, Request
from utils.pdf_parser import extract_text_from_pdf
from utils.chunking import chunk_text
from utils.embedding import embed_texts
from supabase import create_client
import requests, tempfile

app = FastAPI()

# Initialize Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@app.post("/process")
async def process_doc(req: Request):
    try:
        data = await req.json()
        doc_id = data["document_id"]
        file_url = data["file_url"]
        org_id = data["organization_id"]

        print("📥 Received document:", doc_id)
        print("🔗 Downloading from:", file_url)

        # Download PDF
        response = requests.get(file_url)
        response.raise_for_status()
        print("📄 PDF download complete.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        print("📄 PDF saved to:", tmp_path)
        print("🧠 Extracting text...")
        raw_text = extract_text_from_pdf(tmp_path)
        print("✅ Text extracted. Length:", len(raw_text))

        print("🔍 Chunking...")
        chunks = chunk_text(raw_text)
        print("✅ Chunks created:", len(chunks))

        print("🧠 Generating embeddings...")
        vectors = embed_texts(chunks)
        print("✅ Embeddings generated:", len(vectors))

        print("💾 Inserting into Supabase...")
        for content, vector in zip(chunks, vectors):
            supabase.table("document_chunks").insert({
                "document_id": doc_id,
                "organization_id": org_id,
                "content": content,
                "embedding": vector,
                "metadata": {}
            }).execute()
        print("✅ Data inserted into Supabase.")

        print("📡 Sending callback to Supabase...")
        callback_response = requests.post(
            os.getenv("CALLBACK_URL"),
            json={
                "documentId": doc_id,
                "status": "processed"
            },
            headers={"x-api-key": os.getenv("DOCUMENT_PROCESSING_API_KEY")}
        )
        print("✅ Callback sent. Response code:", callback_response.status_code)

        return {"status": "done"}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}
