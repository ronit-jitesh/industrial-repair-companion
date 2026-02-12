#!/usr/bin/env python3
"""
ingest.py
Phase 3: Hybrid RAG Pipeline — Dual FAISS Index Ingestion

Creates two vector indices:
  1. manuals_index — Chunked Siemens VFD manual content
  2. history_index — Embedded repair log Symptom_Strings with metadata
"""

import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ─── Configuration ───────────────────────────────────────────────────────────

MANUALS_DIR = "data/manuals"
CLEANED_CSV = "data/repair_logs_cleaned.csv"
MANUALS_INDEX_DIR = "indices/manuals_index"
HISTORY_INDEX_DIR = "indices/history_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ─── Embedding Model ────────────────────────────────────────────────────────


def load_embeddings():
    """Load HuggingFace sentence-transformer embeddings."""
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("   ✅ Embedding model loaded")
    return embeddings


# ─── Step A: Manuals Index ──────────────────────────────────────────────────


def ingest_manuals(embeddings):
    """
    Load manual text files, chunk them, and build a FAISS index.
    Each chunk retains metadata about source file and section.
    """
    print(f"\n{'='*70}")
    print("📖 STEP A: Ingesting Manuals → manuals_index")
    print(f"{'='*70}")

    # Load all .txt files from manuals directory
    documents = []
    for filename in sorted(os.listdir(MANUALS_DIR)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(MANUALS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract a clean source name
        source_name = filename.replace("_", " ").replace(".txt", "").title()
        documents.append(Document(
            page_content=content,
            metadata={
                "source": source_name,
                "filename": filename,
                "type": "manual",
            }
        ))
        print(f"   📄 Loaded: {filename} ({len(content):,} chars)")

    print(f"\n   Total documents loaded: {len(documents)}")

    # Chunk the documents
    print(f"\n   Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n--------", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # Enrich chunk metadata with section info
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        # Try to extract section info from content
        lines = chunk.page_content.strip().split("\n")
        for line in lines:
            if line.startswith("SECTION") or line.startswith("F3") or line.startswith("F07"):
                chunk.metadata["section"] = line.strip()[:80]
                break

    print(f"   Total chunks: {len(chunks)}")
    print(f"   Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    # Build FAISS index
    print(f"\n   Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index
    os.makedirs(MANUALS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(MANUALS_INDEX_DIR)
    print(f"   ✅ Manuals index saved → {MANUALS_INDEX_DIR}/")

    return vectorstore


# ─── Step B: History Index ──────────────────────────────────────────────────


def ingest_history(embeddings):
    """
    Load cleaned repair logs, embed the Symptom_String column,
    and build a FAISS index with full metadata.
    """
    print(f"\n{'='*70}")
    print("🔧 STEP B: Ingesting Repair History → history_index")
    print(f"{'='*70}")

    # Load cleaned CSV
    df = pd.read_csv(CLEANED_CSV)
    print(f"   📊 Loaded {len(df):,} repair records")

    # Create Document objects from each row
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=str(row["Symptom_String"]),
            metadata={
                "log_id": int(row["Log_ID"]),
                "date": str(row["Date"]),
                "machine_id": str(row["Machine_ID"]),
                "error_code": str(row["Error_Code"]),
                "error_description": str(row["Error_Description"]),
                "operating_temp": float(row["Operating_Temp"]),
                "vibration_level": float(row["Vibration_Level"]),
                "humidity": int(row["Humidity"]),
                "technician_id": str(row["Technician_ID"]),
                "technician_name": str(row["Technician_Name"]),
                "technician_notes": str(row["Technician_Notes"]),
                "site_location": str(row["Site_Location"]),
                "outcome": str(row["Outcome"]),
                "type": "repair_log",
            }
        )
        documents.append(doc)

    print(f"   Created {len(documents):,} document objects")

    # Build FAISS index (embedding in batches for performance)
    print(f"\n   Building FAISS index (this may take a few minutes)...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save index
    os.makedirs(HISTORY_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(HISTORY_INDEX_DIR)
    print(f"   ✅ History index saved → {HISTORY_INDEX_DIR}/")

    return vectorstore


# ─── Test Retrieval ─────────────────────────────────────────────────────────


def test_retrieval(manuals_store, history_store):
    """Run sample queries to verify both indices work."""
    print(f"\n{'='*70}")
    print("🧪 Testing Hybrid Retrieval")
    print(f"{'='*70}")

    test_queries = [
        "Overheating error F30002",
        "Motor encoder signal error vibration",
        "DC link undervoltage brownout",
        "Cooling fan failure maintenance",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")

        # Manual results
        manual_results = manuals_store.similarity_search(query, k=2)
        print(f"\n   📖 Manual Results:")
        for i, doc in enumerate(manual_results):
            source = doc.metadata.get("source", "Unknown")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"      [{i+1}] Source: {source}")
            print(f"          \"{preview}...\"")

        # History results
        history_results = history_store.similarity_search(query, k=3)
        print(f"\n   🔧 History Results:")
        for i, doc in enumerate(history_results):
            log_id = doc.metadata.get("log_id", "?")
            tech = doc.metadata.get("technician_name", "Unknown")
            outcome = doc.metadata.get("outcome", "?")
            notes = doc.metadata.get("technician_notes", "")[:80]
            print(f"      [{i+1}] Log #{log_id} | {tech} | Outcome: {outcome}")
            print(f"          \"{notes}\"")

        print(f"   {'─'*50}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print("🚀 Industrial Repair Companion — Ingestion Pipeline")
    print(f"{'='*70}\n")

    # Load shared embedding model
    embeddings = load_embeddings()

    # Step A: Manuals
    manuals_store = ingest_manuals(embeddings)

    # Step B: History
    history_store = ingest_history(embeddings)

    # Test both indices
    test_retrieval(manuals_store, history_store)

    print(f"\n{'='*70}")
    print("✅ Ingestion Complete!")
    print(f"{'='*70}")
    print(f"   📖 Manuals Index: {MANUALS_INDEX_DIR}/")
    print(f"   🔧 History Index: {HISTORY_INDEX_DIR}/")
    print(f"   🤖 Embedding Model: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
