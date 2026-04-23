# 🤖 TechNova RAG-Based Customer Support Chatbot

> **Innomatics Research Labs | Generative & Agentic AI Internship — Mini Project**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.14-green)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.5-purple)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-orange)](https://groq.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.42.0-red)](https://gradio.app/)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Tech Stack](#-tech-stack)
4. [System Architecture](#-system-architecture)
5. [Project Structure](#-project-structure)
6. [Module Descriptions](#-module-descriptions)
7. [RAG Pipeline — How It Works](#-rag-pipeline--how-it-works)
8. [LangGraph Workflow](#-langgraph-workflow)
9. [Human-in-the-Loop (HITL) Escalation](#-human-in-the-loop-hitl-escalation)
10. [Setup & Installation](#-setup--installation)
11. [Running the Application](#-running-the-application)
12. [Example Interactions](#-example-interactions)
13. [Internship Context](#-internship-context)

---

## 📌 Project Overview

This project is a **Retrieval-Augmented Generation (RAG) powered customer support chatbot** built for a fictional company called **TechNova Solutions**. It was developed as a mini project during the **Generative and Agentic AI Internship at Innomatics Research Labs**.

The chatbot is capable of answering customer queries by retrieving relevant information from a structured knowledge base (PDF), generating context-aware responses using a large language model (LLM), and automatically escalating complex or urgent queries to a human support agent via a **Human-in-the-Loop (HITL)** mechanism.

The entire workflow is orchestrated using **LangGraph**, a state-based agentic framework built on top of LangChain, making the pipeline modular, traceable, and production-ready.

---

## ✨ Key Features

- **RAG Pipeline** — Answers are grounded in a domain-specific knowledge base (PDF), eliminating hallucinations.
- **LangGraph Orchestration** — A 3-node agentic state graph manages query processing, answer generation, and escalation.
- **ChromaDB Vector Store** — All document chunks are stored as dense vector embeddings for efficient semantic search.
- **Groq LLM (LLaMA 3.1)** — Ultra-fast inference using the `llama-3.1-8b-instant` model via the Groq API.
- **HuggingFace Embeddings** — Uses `all-MiniLM-L6-v2` for lightweight, high-quality sentence embeddings (runs on CPU).
- **HITL Escalation** — Detects urgent/complex queries (e.g., fraud, legal, refund disputes) and routes them to a human agent, logging them to `escalation_log.txt`.
- **Gradio Web UI** — Clean, interactive browser-based chat interface with example queries.

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Groq — LLaMA 3.1 8B Instant | Answer generation |
| **Orchestration** | LangGraph 0.2.14 | Agentic state machine workflow |
| **Vector Store** | ChromaDB 0.5.5 | Semantic chunk storage & retrieval |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Text → vector conversion |
| **Document Loader** | PyPDF + LangChain | PDF ingestion & text extraction |
| **Text Splitter** | RecursiveCharacterTextSplitter | Chunking with overlap |
| **Web UI** | Gradio 4.42.0 | Frontend chat interface |
| **Language** | Python 3.10+ | Core programming language |
| **Environment** | python-dotenv | API key management |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER (Browser)                       │
│                   Gradio Web UI                         │
└──────────────────────┬──────────────────────────────────┘
                       │  user query
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  app.py (Entry Point)                   │
│              Gradio Interface Handler                   │
└──────────────────────┬──────────────────────────────────┘
                       │  calls run_graph(query)
                       ▼
┌─────────────────────────────────────────────────────────┐
│               graph.py (LangGraph Engine)               │
│                                                         │
│  ┌─────────────┐     ┌──────────────┐  ┌────────────┐  │
│  │Node 1       │────▶│ Node 2       │  │ Node 3     │  │
│  │process_query│     │generate_answer│  │escalate_to │  │
│  │             │────▶│              │  │_human      │  │
│  └─────────────┘     └──────────────┘  └────────────┘  │
│         │                                               │
│  retrieve chunks                                        │
│  check HITL                                             │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐    ┌────────────────────────┐
│  retriever.py    │    │       hitl.py           │
│  ChromaDB query  │    │  Escalation detection   │
│  → top-k chunks  │    │  & ticket logging       │
└──────────────────┘    └────────────────────────┘
          │
          ▼
┌──────────────────┐
│  chroma_db/      │
│  (Vector Store)  │
│  ← ingest.py     │
└──────────────────┘
```

---

## 📁 Project Structure

```
Innomatics_Internship_MIni_Project-RAG-BASED-CHATBOT-/
│
├── app.py                  # Gradio web UI — main entry point
├── graph.py                # LangGraph state machine (3 nodes)
├── ingest.py               # PDF ingestion pipeline (PDF → ChromaDB)
├── retriever.py            # ChromaDB semantic retriever
├── hitl.py                 # Human-in-the-Loop escalation manager
├── test_env.py             # Environment & dependency check script
│
├── knowledge_base.pdf      # TechNova domain knowledge source
├── escalation_log.txt      # Auto-generated escalation ticket log
├── requirements.txt        # All Python dependencies
├── .gitignore              # Git ignore rules
│
└── chroma_db/              # Persisted ChromaDB vector database
    └── ...                 # (auto-generated after running ingest.py)
```

---

## 📦 Module Descriptions

### `ingest.py` — PDF Ingestion Pipeline

Responsible for the one-time setup of the vector store. It executes a 4-step pipeline:

1. **Load PDF** — Uses `PyPDFLoader` to read all pages from `knowledge_base.pdf`.
2. **Chunk Documents** — Splits text into overlapping chunks (`chunk_size=500`, `chunk_overlap=50`) using `RecursiveCharacterTextSplitter`.
3. **Create Embeddings** — Loads the `all-MiniLM-L6-v2` HuggingFace model to convert text chunks to dense vectors.
4. **Store in ChromaDB** — Persists all chunk vectors to the local `./chroma_db` directory.

> Run this **once** before starting the application: `python ingest.py`

---

### `retriever.py` — Semantic Retriever

Connects to the persisted ChromaDB vector store and performs a **similarity search** for a given user query. It returns the top-k most semantically relevant text chunks, which are then passed as context to the LLM.

---

### `hitl.py` — Human-in-the-Loop Manager

The `HITLManager` class implements **automatic escalation logic**. It scans queries for trigger keywords (e.g., `fraud`, `refund`, `urgent`, `legal`, `lawyer`) and conditions like very short or ambiguous context. If escalation is warranted, it:

- Generates a structured support ticket.
- Appends the ticket to `escalation_log.txt` with a timestamp.
- Returns a user-facing message confirming escalation.

---

### `graph.py` — LangGraph Workflow Engine

The core orchestration file. Defines a **typed state schema** and three functional nodes connected via conditional edges:

| Node | Responsibility |
|---|---|
| `process_query` | Retrieves chunks from ChromaDB, checks HITL conditions, sets route |
| `generate_answer` | Calls Groq LLM with retrieved context, stores answer in state |
| `escalate_to_human` | Calls HITLManager to log ticket and generate escalation message |

The graph is compiled and exported as `rag_graph`. The public function `run_graph(query)` accepts a user query and returns the final state dictionary containing the answer and escalation status.

---

### `app.py` — Gradio Web Interface

Wraps the entire system in a browser-based UI. Key components:

- A multi-line text input for user queries.
- An output box showing the formatted AI or escalation response.
- Pre-loaded **example questions** for quick testing.
- An info section explaining how the system works.
- Runs on `localhost:7860` by default.

---

## 🔄 RAG Pipeline — How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  1. RETRIEVE                        │
│  Query is embedded using            │
│  all-MiniLM-L6-v2 and compared      │
│  against ChromaDB vectors.          │
│  Top-k relevant chunks returned.    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. AUGMENT                         │
│  Retrieved chunks are injected      │
│  into the LLM prompt as context.    │
│  A structured prompt template       │
│  is assembled.                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. GENERATE                        │
│  Groq LLM (LLaMA 3.1 8B) generates │
│  a grounded answer using only the   │
│  provided context. If context is    │
│  insufficient, a fallback message   │
│  with the support number is shown.  │
└─────────────────────────────────────┘
```

---

## 🔀 LangGraph Workflow

The state graph follows this flow:

```
START
  │
  ▼
[process_query]  ←── Node 1
  │
  ├── route == "answer"   ──▶  [generate_answer]   ──▶  END
  │
  └── route == "escalate" ──▶  [escalate_to_human] ──▶  END
```

The `State` TypedDict carries the following fields through the graph:

```python
class State(TypedDict):
    query      : str        # User's input question
    chunks     : List[str]  # Retrieved context chunks
    answer     : str        # Final response to return
    route      : str        # "answer" or "escalate"
    escalated  : bool       # True if routed to human agent
    session_id : str        # Unique ID for this interaction
```

---

## 👨‍💼 Human-in-the-Loop (HITL) Escalation

The HITL system is a deliberate design choice to make the chatbot **production-safe**. Not every query should be answered by an AI — some situations require human empathy and authority.

**Escalation is triggered when the query contains keywords such as:**

- Urgency indicators: `urgent`, `immediately`, `asap`
- Legal/financial: `fraud`, `lawsuit`, `legal`, `lawyer`, `sue`
- Sensitive service: `refund dispute`, `data breach`, `account hacked`
- Weak context: fewer than 2 relevant chunks retrieved

**On escalation, the system:**

1. Logs a structured ticket to `escalation_log.txt` with timestamp, session ID, and query.
2. Informs the user that their query has been flagged and a human agent will follow up.
3. Provides the direct support contact number (`1800-123-4567`).

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or higher
- A [Groq API key](https://console.groq.com/) (free tier available)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/AkshatMishra29/Innomatics_Internship_MIni_Project-RAG-BASED-CHATBOT-.git
cd Innomatics_Internship_MIni_Project-RAG-BASED-CHATBOT-
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the HuggingFace embedding model (~90 MB). This is a one-time download.

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Never commit your `.env` file. It is already listed in `.gitignore`.

### Step 5 — Ingest the Knowledge Base

Run the ingestion pipeline once to populate ChromaDB:

```bash
python ingest.py
```

Expected output:
```
==================================================
 RAG INGESTION PIPELINE STARTED
==================================================
📄 Step 1: Loading PDF ...       ✅ Loaded N pages.
✂️  Step 2: Splitting chunks ... ✅ Created N chunks.
🧠 Step 3: Loading embeddings ... ✅ Model loaded.
💾 Step 4: Storing in ChromaDB ... ✅ Stored N chunks.
==================================================
 ✅ INGESTION COMPLETE!
==================================================
```

---

## 🚀 Running the Application

```bash
python app.py
```

The Gradio interface will launch at:

```
🌐 http://localhost:7860
```

Open the URL in your browser to start interacting with the chatbot.

---

## 💬 Example Interactions

| Query | Expected Behaviour |
|---|---|
| `What is the price of SmartHome Hub?` | AI answers from knowledge base |
| `How do I return a product?` | AI provides return policy |
| `My camera live feed is not loading` | AI provides troubleshooting steps |
| `What payment methods are accepted?` | AI answers from knowledge base |
| `I want a refund, this is fraud!` | 🔴 **Escalated to human agent** |
| `This is urgent, I will take legal action` | 🔴 **Escalated to human agent** |

---

## 🎓 Internship Context

| Detail | Information |
|---|---|
| **Organization** | Innomatics Research Labs |
| **Internship Track** | Generative and Agentic AI |
| **Project Type** | Mini Project |
| **Intern** | Akshat Mishra |
| **Repository** | [GitHub Link](https://github.com/AkshatMishra29/Innomatics_Internship_MIni_Project-RAG-BASED-CHATBOT-) |

### Concepts Demonstrated

This project demonstrates practical, hands-on implementation of the following concepts covered during the internship:

- **Retrieval-Augmented Generation (RAG)** — Grounding LLM responses in private/domain-specific documents.
- **Agentic AI with LangGraph** — Building stateful, multi-node AI workflows with conditional routing.
- **Vector Databases** — Using ChromaDB for persistent semantic search over embedded document chunks.
- **LLM Integration** — Connecting to the Groq API for fast inference with open-source LLaMA models.
- **Human-in-the-Loop (HITL)** — Designing safe AI systems that know when to defer to a human.
- **Prompt Engineering** — Crafting structured prompts that produce reliable, context-aware responses.
- **Production UI** — Wrapping an AI backend in a clean, shareable Gradio web interface.

---

## 📄 Dependencies

```
langchain==0.2.16
langchain-community==0.2.16
langchain-groq==0.1.9
langgraph==0.2.14
chromadb==0.5.5
sentence-transformers==3.0.1
pypdf==4.3.1
gradio==4.42.0
python-dotenv==1.0.1
```

---

## 🔒 Security Notes

- The `.env` file containing the `GROQ_API_KEY` is excluded from version control via `.gitignore`.
- Do not hard-code API keys anywhere in the source code.
- The `chroma_db/` directory contains only vector embeddings of the knowledge base — no sensitive user data is persisted.

---

*Built with ❤️ during the Innomatics Generative & Agentic AI Internship*
