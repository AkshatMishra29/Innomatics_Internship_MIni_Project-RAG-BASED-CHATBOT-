

# ============================================================
# app.py — Gradio Web Interface
# Connects everything together into a chat UI
# ============================================================

import gradio as gr
from graph import run_graph


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CHAT FUNCTION
# Called every time user clicks Ask Question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chat(query: str) -> str:
    """
    Takes user query, runs through LangGraph,
    returns formatted answer.
    """

    # Empty query check
    if not query.strip():
        return "⚠️ Please type a question first."

    try:
        # Run full RAG graph workflow
        result = run_graph(query)

        # Format response based on escalation status
        if result["escalated"]:
            response = "👨‍💼 ESCALATED TO HUMAN AGENT\n"
            response += "─" * 40 + "\n"
            response += result["answer"]
        else:
            response = "🤖 AI ANSWER\n"
            response += "─" * 40 + "\n"
            response += result["answer"]

        return response

    except Exception as e:
        return f"❌ Error: {str(e)}\nPlease try again or call 1800-123-4567."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRADIO INTERFACE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with gr.Blocks(title="TechNova Support Assistant") as demo:

    # ─── HEADER ──────────────────────────────────────────────
    gr.Markdown("""
    # 🤖 TechNova RAG Customer Support Assistant
    ### Powered by LangGraph + ChromaDB + Groq LLM
    ---
    Ask any question about our products, billing, returns, or technical support.
    Complex or urgent queries are automatically escalated to a human agent.
    """)

    # ─── MAIN CHAT AREA ──────────────────────────────────────
    with gr.Row():

        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                lines=3
            )

            ask_btn = gr.Button(
                "Ask Question 🚀",
                variant="primary"
            )

            clear_btn = gr.Button(
                "Clear",
                variant="secondary"
            )

        with gr.Column(scale=3):
            output_box = gr.Textbox(
                label="Support Response",
                lines=12,
                interactive=False
            )

    # ─── EXAMPLE QUESTIONS ───────────────────────────────────
    gr.Markdown("### 💡 Example Questions — Click to Try")
    gr.Examples(
        examples=[
            ["What is the price of SmartHome Hub?"],
            ["How do I return a product?"],
            ["My camera live feed is not loading"],
            ["What payment methods are accepted?"],
            ["How do I reset my SmartLock fingerprint?"],
            ["I want a refund urgently, this is fraud!"],
            ["What is covered under warranty?"],
            ["How do I delete my account?"],
        ],
        inputs=input_box
    )

    # ─── INFO SECTION ─────────────────────────────────────────
    gr.Markdown("""
    ---
    ### ℹ️ How It Works
    - 🔍 Your question is searched against the TechNova knowledge base
    - 🧠 Relevant information is retrieved from ChromaDB
    - 🤖 Groq LLM generates a precise answer using that context
    - 👨‍💼 Urgent or complex queries are escalated to a human agent

    📞 **Direct Support:** 1800-123-4567 | ✉️ support@technova.in
    """)

    # ─── BUTTON ACTIONS ──────────────────────────────────────
    ask_btn.click(
        fn=chat,
        inputs=input_box,
        outputs=output_box
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[input_box, output_box]
    )

    # Also trigger on Enter key
    input_box.submit(
        fn=chat,
        inputs=input_box,
        outputs=output_box
    )


# ─── LAUNCH ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   LAUNCHING TECHNOVA SUPPORT ASSISTANT")
    print("=" * 50)
    print("\n🌐 Opening at: http://localhost:7860\n")

    demo.launch(
        server_port=7860,
        share=False,
        show_error=True
    )