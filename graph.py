# ============================================================
# graph.py — LangGraph Workflow Engine
# 3 Nodes: process_query → generate_answer / escalate_to_human
# ============================================================

import os
import uuid
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from retriever import retrieve_chunks
from hitl import HITLManager

# Load .env file
load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATE — flows through every node in the graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class State(TypedDict):
    query      : str        # user question
    chunks     : List[str]  # retrieved chunks
    answer     : str        # final answer
    route      : str        # "answer" or "escalate"
    escalated  : bool       # True if sent to human
    session_id : str        # unique session ID


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 1 — process_query
# Retrieves chunks and decides routing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_query(state: State) -> State:
    """
    Node 1: Retrieve relevant chunks and decide route.
    - Calls ChromaDB retriever
    - Checks HITL escalation conditions
    - Sets state route to answer or escalate
    """
    print("\n" + "─" * 45)
    print("⚙️  NODE 1: process_query")
    print("─" * 45)

    query = state["query"]
    print(f"   Query: {query}")

    # Retrieve chunks from ChromaDB
    chunks = retrieve_chunks(query)
    state["chunks"] = chunks

    # Check if escalation is needed
    hitl = HITLManager()
    should_escalate = hitl.check_escalation(query, chunks)

    if should_escalate:
        state["route"] = "escalate"
        print("   🔀 Route: ESCALATE TO HUMAN")
    else:
        state["route"] = "answer"
        print("   🔀 Route: GENERATE AI ANSWER")

    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 2 — generate_answer
# Uses Groq LLM to answer using retrieved chunks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_answer(state: State) -> State:
    """
    Node 2: Generate AI answer using context chunks.
    - Builds prompt with retrieved chunks
    - Calls Groq LLM
    - Stores answer in state
    """
    print("\n" + "─" * 45)
    print("🤖 NODE 2: generate_answer")
    print("─" * 45)

    query  = state["query"]
    chunks = state["chunks"]

    # Join all chunks into one context block
    context = "\n\n".join(chunks)

    # Build prompt
    prompt = f"""You are a helpful customer support assistant for TechNova Solutions.
Use the context below to answer the customer question accurately.
If the context does not contain the answer, say:
"I don't have enough information to answer this. Please contact support at 1800-123-4567."

Context:
{context}

Customer Question:
{query}

Answer:"""

    try:
        print("   📡 Calling Groq LLM ...")
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )
        response = llm.invoke(prompt)
        state["answer"]    = response.content
        state["escalated"] = False
        print("   ✅ Answer generated.")

    except Exception as e:
        print(f"   ❌ LLM ERROR: {e}")
        state["answer"]    = "Sorry, I am unable to process your request right now. Please try again or call 1800-123-4567."
        state["escalated"] = False

    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 3 — escalate_to_human
# Creates HITL ticket and returns escalation message
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def escalate_to_human(state: State) -> State:
    """
    Node 3: Escalate query to human agent.
    - Creates support ticket via HITLManager
    - Stores escalation message in state
    """
    print("\n" + "─" * 45)
    print("👨‍💼 NODE 3: escalate_to_human")
    print("─" * 45)

    hitl = HITLManager()
    message = hitl.handle_escalation(state["query"])

    state["answer"]    = message
    state["escalated"] = True

    print("   ✅ Escalation complete.")
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTING FUNCTION
# Reads state["route"] and returns next node name
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def decide_route(state: State) -> str:
    """
    Conditional edge function.
    Returns name of next node based on route in state.
    """
    if state["route"] == "escalate":
        return "escalate_to_human"
    return "generate_answer"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD THE GRAPH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Create graph with State as schema
graph_builder = StateGraph(State)

# Add all 3 nodes
graph_builder.add_node("process_query",      process_query)
graph_builder.add_node("generate_answer",    generate_answer)
graph_builder.add_node("escalate_to_human",  escalate_to_human)

# Set entry point
graph_builder.set_entry_point("process_query")

# Add conditional edge from Node 1
graph_builder.add_conditional_edges(
    "process_query",
    decide_route,
    {
        "generate_answer"   : "generate_answer",
        "escalate_to_human" : "escalate_to_human"
    }
)

# Both Node 2 and Node 3 lead to END
graph_builder.add_edge("generate_answer",   END)
graph_builder.add_edge("escalate_to_human", END)

# Compile the graph
rag_graph = graph_builder.compile()

print("✅ LangGraph compiled successfully.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN FUNCTION — called by app.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_graph(query: str) -> dict:
    """
    Main function to run the full RAG workflow.

    Args:
        query (str): User question

    Returns:
        dict: Final state with answer and escalation status
    """
    print("\n" + "=" * 45)
    print("   RAG GRAPH STARTED")
    print("=" * 45)

    # Create initial state
    initial_state: State = {
        "query"      : query,
        "chunks"     : [],
        "answer"     : "",
        "route"      : "",
        "escalated"  : False,
        "session_id" : str(uuid.uuid4())[:8]
    }

    # Run the graph
    final_state = rag_graph.invoke(initial_state)

    print("\n" + "=" * 45)
    print("   RAG GRAPH COMPLETE")
    print("=" * 45)

    return final_state


# ─── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n--- TEST 1: Normal Question ---")
    result1 = run_graph("What is the price of SmartHome Hub?")
    print(f"\n📢 Answer:\n{result1['answer']}")
    print(f"   Escalated: {result1['escalated']}")

    print("\n\n--- TEST 2: Escalation Question ---")
    result2 = run_graph("I want a refund, this is fraud!")
    print(f"\n📢 Answer:\n{result2['answer']}")
    print(f"   Escalated: {result2['escalated']}")