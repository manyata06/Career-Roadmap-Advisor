# Career Roadmap Advisor — v2 (TF-IDF + RAG)

A Streamlit app that matches users to career paths using either keyword-based TF-IDF similarity or semantic RAG search powered by ChromaDB and sentence-transformers.

---

## What's new in v2

| Feature | v1 | v2 |
|---|---|---|
| Search | TF-IDF keyword matching | TF-IDF **+** RAG semantic search |
| Understands vague queries | No | Yes (RAG mode) |
| Vector store | None | ChromaDB (in-memory) |
| Embeddings | None | all-MiniLM-L6-v2 (local, free) |
| API key needed | No | No |
| RAG transparency panel | No | Yes |

---

## How to run locally

### 1. Clone / download this folder

```bash
cd career-advisor-v2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` model (~90MB). After that it's cached locally.

### 3. Run the app

```bash
streamlit run app.py
```

---

## How RAG works in this app

```
User types query
       ↓
all-MiniLM-L6-v2 converts it to a 384-dimension vector
       ↓
ChromaDB searches in-memory store of career vectors
       ↓
Top-N most similar careers returned (cosine similarity)
       ↓
Results displayed with transparency panel showing what was retrieved
```

### Why RAG is better than TF-IDF for vague queries

| Query | TF-IDF finds | RAG finds |
|---|---|---|
| "I like coding and AI" | ML Engineer, Data Scientist | ML Engineer, AI Researcher, Data Scientist |
| "I want to help people think clearly" | Nothing (no keyword match) | Coach, Therapist, UX Researcher |
| "creative but analytical" | Nothing | UX Designer, Data Journalist, Strategist |

---

## Project structure

```
career-advisor-v2/
├── app.py                 # Main Streamlit app (TF-IDF + RAG)
├── career_interest.csv    # Career dataset (261 careers)
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## Skills demonstrated (relevant for ML internship applications)

- **RAG pipeline**: ChromaDB + sentence-transformers
- **Prompt-less LLM pattern**: semantic search without an API key
- **Streamlit deployment**: interactive UI with caching
- **NLP**: TF-IDF, cosine similarity, embedding models
- **Python**: pandas, scikit-learn, chromadb
