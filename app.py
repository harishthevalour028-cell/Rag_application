import os

import streamlit as st
from dotenv import load_dotenv

from rag import (
    answer_question,
    build_vectorstore,
    get_embeddings,
    load_documents,
)

load_dotenv()

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")


def get_api_key() -> str:
    try:
        secret = st.secrets.get("GROQ_API_KEY")
    except (FileNotFoundError, KeyError):
        secret = None
    return (secret or os.getenv("GROQ_API_KEY", "")).strip()


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []


initialize_state()

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    process_documents = st.button(
        "Process documents", type="primary", use_container_width=True
    )
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Higher values require a closer semantic match.",
    )

    st.subheader("Indexed files")
    if st.session_state.indexed_files:
        for filename in st.session_state.indexed_files:
            st.caption(f"✓ {filename}")
    else:
        st.caption("No documents indexed yet.")

    if st.button("Clear", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.indexed_files = []
        st.session_state.messages = []
        st.rerun()

if process_documents:
    if not uploaded_files:
        st.sidebar.error("Upload at least one PDF, DOCX, or TXT file first.")
    else:
        with st.spinner("Reading and indexing documents..."):
            try:
                documents, filenames = load_documents(uploaded_files)
                st.session_state.vectorstore = build_vectorstore(
                    documents, get_embeddings()
                )
                st.session_state.indexed_files = filenames
                st.session_state.messages = []
                st.sidebar.success(f"Indexed {len(filenames)} file(s).")
            except ValueError as exc:
                st.sidebar.error(str(exc))
            except Exception as exc:
                st.sidebar.error(f"Could not index documents: {exc}")

st.title("📚 Document Q&A")
st.write("Upload documents in the sidebar, index them, and ask questions about their content.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Source chunks"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['file']}** — page {source['page']} "
                        f"(score: {source['score']:.2f})"
                    )
                    st.caption(source["text"])

question = st.chat_input("Ask a question about your documents")
if question:
    if st.session_state.vectorstore is None:
        st.warning("Process at least one document before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and answering..."):
                try:
                    response, sources = answer_question(
                        st.session_state.vectorstore,
                        question,
                        threshold,
                        get_api_key(),
                    )
                except ValueError as exc:
                    response, sources = str(exc), []
                except Exception as exc:
                    error_text = str(exc)
                    if "404" in error_text:
                        response = (
                            "Groq could not find the configured model. "
                            "The app's model may have been retired; update the "
                            "GROQ_MODEL setting in rag.py."
                        )
                    else:
                        response = f"Could not answer the question: {error_text}"
                    sources = []
            st.markdown(response)
            if sources:
                with st.expander("Source chunks"):
                    for source in sources:
                        st.markdown(
                            f"**{source['file']}** — page {source['page']} "
                            f"(score: {source['score']:.2f})"
                        )
                        st.caption(source["text"])

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "sources": sources}
        )
