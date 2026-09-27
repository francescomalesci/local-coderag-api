import os
from pypdf import PdfReader
from database import collection

DOCS_DIR = "./documents"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into chunks with a slight overlap to preserve context."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def process_pdf(file_path: str) -> str:
    """Extracts text from a given PDF file."""
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def main():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"[SYSTEM] Created directory '{DOCS_DIR}'. Add PDF or TXT files and run again.")
        return

    print("[SYSTEM] Starting document scanning pipeline...")

    processed_files = 0
    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        full_text = ""
        
        if filename.endswith(".pdf"):
            full_text = process_pdf(file_path)
        elif filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        else:
            print(f"[WARNING] Unsupported format, skipping: {filename}")
            continue
            
        if not full_text.strip():
            print(f"[WARNING] Empty file, skipping: {filename}")
            continue
            
        # Apply chunking
        chunks = chunk_text(full_text)
        
        docs = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})
            ids.append(f"{filename}_chunk_{i}")
            
        # Save to ChromaDB via upsert to prevent duplicate ID crashes
        if docs:
            collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
            print(f"[SUCCESS] Ingested: {filename} ({len(chunks)} vectorized chunks).")
            processed_files += 1

    print(f"\n[SYSTEM] Pipeline completed: {processed_files} files added to the Vector DB.")

if __name__ == "__main__":
    main()