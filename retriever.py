

# ============================================================
# retriever.py — Vector Search Module
# Loads ChromaDB → Takes query → Returns matching chunks
# ============================================================

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─── SETTINGS ────────────────────────────────────────────────
CHROMA_DIR      = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 3


def load_vectorstore():
    print("🔌 Loading ChromaDB ...")
    try:
        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(
                "chroma_db folder not found. Run ingest.py first."
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        print("✅ ChromaDB loaded.")
        return vectorstore

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        raise
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise


def retrieve_chunks(query: str, top_k: int = TOP_K) -> list:
    """
    Search ChromaDB for most relevant chunks for a query.

    Args:
        query  (str): User question
        top_k  (int): Number of chunks to return

    Returns:
        list: Relevant text chunks. Empty list if error.
    """
    print(f"\n🔍 Searching for: '{query}'")
    try:
        vectorstore = load_vectorstore()
        results = vectorstore.similarity_search(query, k=top_k)

        if not results:
            print("   ⚠️  No chunks found.")
            return []

        chunks = [doc.page_content for doc in results]
        print(f"   ✅ Found {len(chunks)} relevant chunks.")

        for i, chunk in enumerate(chunks):
            preview = chunk[:80].replace("\n", " ")
            print(f"   Chunk {i+1}: {preview}...")

        return chunks

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return []


# ─── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   RETRIEVER TEST")
    print("=" * 50)

    query  = "How do I return a product?"
    chunks = retrieve_chunks(query)

    print(f"\n📋 Results for: '{query}'")
    print("-" * 50)
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:\n{chunk}")
        print("-" * 50)