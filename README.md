# Minimal LangChain RAG app

A small document question-answering app using LangChain LCEL, Streamlit, FastEmbed
embeddings, and an in-memory FAISS index. It does not use PyTorch.

## Run locally

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GROQ_API_KEY
streamlit run app.py
```

The first embedding-model load downloads the ONNX model and can take a little
longer. The model is cached for the rest of the process.

## Streamlit Community Cloud

Deploy this repository with `app.py` as the main file. Add `GROQ_API_KEY` under
**Settings > Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Uploaded files and the FAISS index live only in session memory. They are cleared
with the **Clear** button or when the app session ends.

## How it works

1. Uploaded files are saved temporarily and loaded with the appropriate
   LangChain loader.
2. Text is split into 500-character chunks with 80-character overlap.
3. normalized `BAAI/bge-small-en-v1.5` FastEmbed vectors are stored in an
   in-memory FAISS index.
4. Each question retrieves four chunks and converts normalized-vector distance to
   a 0-1 relevance score. Questions below the selected threshold never call Groq.
5. Otherwise, an LCEL chain sends only the retrieved context and current question
   to `ChatGroq` using Groq's current `openai/gpt-oss-20b` model, with source
   file and page metadata shown below the answer.

## Test questions

Upload a document containing the facts needed for the first three questions:

1. What is the document's main purpose?
2. What date, version, or named project does it mention?
3. What is one specific fact or instruction stated in the document?

These should be answerable from matching document content. Then ask:

4. What is the population of Japan?
5. Who won the latest World Cup?

For documents that do not contain those facts, the last two should return:
`I don't know based on the provided documents.`

## Evaluator explanation

“The app temporarily loads PDF, DOCX, and TXT files, splits their text into
overlapping chunks, and embeds those chunks with a lightweight ONNX model.
Normalized embeddings are indexed in an in-memory FAISS store. For every question
the app retrieves the four closest chunks and applies a relevance threshold before
calling the Groq LLM. The LCEL prompt gives the model only those chunks, requires
source citations, and forbids outside knowledge. Streamlit keeps the index and
chat history in session state, so the app stays lightweight and deployable on the
1 GB Community Cloud runtime.”
