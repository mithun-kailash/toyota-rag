"""Streamlit chat page using the lite (BM25) engine.   Run locally:  streamlit run streamlit_app.py"""
import os

import streamlit as st

import rag_lite as r

st.set_page_config(page_title="Toyota Knowledge Base Chat", page_icon="🚗")
st.title("🚗 Toyota Knowledge Base Chat")
st.caption("Answers come only from the Toyota Cars Complete Knowledge Base PDF.")

# API key: Streamlit secrets (when deployed) or .env / environment variable (on your laptop)
try:
    for name in ("GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        if name in st.secrets:
            os.environ[name] = st.secrets[name]
except Exception:
    pass
r.load_env()
use_llm = r.llm_available()


@st.cache_resource
def load_engine():
    return r.BM25(r.build_chunks())


engine = load_engine()

with st.sidebar:
    k = st.slider("Passages to retrieve", 2, 8, 5)
    st.write("Mode: " + ("AI writes answers" if use_llm else "showing matching passages only"))
    st.markdown("**Try asking**")
    for q in [
        "What is the difference between Innova Crysta and Innova Hycross?",
        "Do conventional Toyota hybrids need to be plugged in?",
        "What do Just-in-Time and Jidoka mean?",
        "Which Toyota models are listed for India?",
    ]:
        if st.button(q):
            st.session_state["pending"] = q

if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask about Toyota...") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        results = engine.search(question, k)
        if not results:
            answer = "I could not find that in the knowledge base."
        elif use_llm:
            try:
                answer = r.ask_llm(question, results)
            except Exception as e:
                answer = f"Could not get an AI answer ({e}). The matching passages are below."
        else:
            answer = "Here are the best matching passages from the knowledge base:"
        st.markdown(answer)
        with st.expander("Sources", expanded=not use_llm):
            for n, res in enumerate(results, start=1):
                st.markdown(f"**[{n}] {res['section']}** · page {', '.join(map(str, res['pages']))}")
                st.text(res["text"])
    st.session_state.messages.append({"role": "assistant", "content": answer})
