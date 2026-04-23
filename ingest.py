# ============================================================
# ingest.py — PDF Ingestion Pipeline
# PDF → Chunks → Embeddings → ChromaDB
# ============================================================

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─── SETTINGS ────────────────────────────────────────────────
PDF_PATH        = "knowledge_base.pdf"
CHROMA_DIR      = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50


def load_pdf(pdf_path):
    print(f"\n📄 Step 1: Loading PDF from '{pdf_path}' ...")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"   ✅ Loaded {len(documents)} pages.")
        return documents
    except FileNotFoundError:
        print("   ❌ ERROR: PDF file not found.")
        print("   Make sure knowledge_base.pdf is in the same folder.")
        raise
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        raise


def chunk_documents(documents):
    print(f"\n✂️  Step 2: Splitting into chunks ...")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"   ✅ Created {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        raise


def create_embeddings():
    print(f"\n🧠 Step 3: Loading embedding model ...")
    print("   (First run downloads ~90MB model, please wait)")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("   ✅ Embedding model loaded.")
        return embeddings
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        raise


def store_in_chromadb(chunks, embeddings):
    print(f"\n💾 Step 4: Storing in ChromaDB ...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        print(f"   ✅ Stored {len(chunks)} chunks in ChromaDB.")
        print(f"   📁 Saved at: {os.path.abspath(CHROMA_DIR)}")
        return vectorstore
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        raise


def ingest_pdf(pdf_path=PDF_PATH):
    print("=" * 50)
    print("   RAG INGESTION PIPELINE STARTED")
    print("=" * 50)

    documents  = load_pdf(pdf_path)
    chunks     = chunk_documents(documents)
    embeddings = create_embeddings()
    store_in_chromadb(chunks, embeddings)

    print("\n" + "=" * 50)
    print("   ✅ INGESTION COMPLETE!")
    print("=" * 50)


# ─── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ingest_pdf()