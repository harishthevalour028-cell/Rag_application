import math
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

GROQ_MODEL = "openai/gpt-oss-20b"


@st.cache_resource
def get_embeddings() -> FastEmbedEmbeddings:
    return NormalizedFastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
    )


class NormalizedFastEmbedEmbeddings(FastEmbedEmbeddings):
    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._normalize(vector) for vector in super().embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._normalize(super().embed_query(text))


def _loader_for(path: Path):
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
    }
    try:
        loader = loaders[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported file type: {path.suffix}") from exc
    return loader(str(path))


def load_documents(uploaded_files: list[Any]) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    filenames: list[str] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            if not uploaded_file.getvalue():
                raise ValueError(f"'{uploaded_file.name}' is empty.")

            path = Path(temp_dir) / Path(uploaded_file.name).name
            path.write_bytes(uploaded_file.getvalue())
            loaded = _loader_for(path).load()
            extracted_text = "\n".join(doc.page_content.strip() for doc in loaded).strip()
            if not extracted_text:
                if path.suffix.lower() == ".pdf":
                    raise ValueError(
                        f"'{uploaded_file.name}' has no extractable text. "
                        "It may be a scanned PDF; upload a text-based PDF."
                    )
                raise ValueError(f"'{uploaded_file.name}' has no extractable text.")

            for document in loaded:
                document.metadata["source"] = uploaded_file.name
            documents.extend(loaded)
            filenames.append(uploaded_file.name)

    if not documents:
        raise ValueError("No extractable text was found in the uploaded documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks could be created from the uploaded documents.")
    return chunks, filenames


def build_vectorstore(
    documents: list[Document], embeddings: FastEmbedEmbeddings
) -> FAISS:
    return FAISS.from_documents(documents, embeddings, normalize_L2=True)


def _page_for(document: Document) -> str:
    page = document.metadata.get("page")
    return str(page + 1) if isinstance(page, int) else "N/A"


def _source_chunks(vectorstore: FAISS, question: str) -> tuple[list[Document], list[dict]]:
    matches = vectorstore.similarity_search_with_score(question, k=4)
    sources: list[dict] = []
    for document, distance in matches:
        relevance = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
        sources.append(
            {
                "file": document.metadata.get("source", "Unknown file"),
                "page": _page_for(document),
                "score": relevance,
                "text": document.page_content,
            }
        )
    return [document for document, _ in matches], sources


def answer_question(
    vectorstore: FAISS, question: str, threshold: float, api_key: str
) -> tuple[str, list[dict]]:
    documents, sources = _source_chunks(vectorstore, question)
    if not sources or sources[0]["score"] < threshold:
        return "I couldn't find this in the uploaded documents.", sources
    if not api_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Add it to Streamlit secrets or your local .env file."
        )

    context = "\n\n".join(
        f"Source file: {source['file']}, page: {source['page']}\n{document.page_content}"
        for document, source in zip(documents, sources)
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer ONLY using the provided context. If the answer is not in the "
                "context, reply exactly: 'I don't know based on the provided documents.' "
                "Never use outside knowledge. Cite the source file and page for each claim.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=api_key,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}), sources
