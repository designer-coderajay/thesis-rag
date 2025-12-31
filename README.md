# 📚 Thesis RAG System

A Retrieval-Augmented Generation (RAG) system for academic research, built for my Master's thesis on **Explainable AI for LLMs**.

## 🎯 What This Does

- **Indexes research papers** (PDFs, web articles, YouTube transcripts)
- **Semantic search** across 50+ papers using FAISS embeddings
- **AI-powered Q&A** with proper source citations
- **Literature review assistance** with real metadata extraction

## 🛠️ Tech Stack

| Component    | Technology                            |
| ------------ | ------------------------------------- |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Embeddings   | `nomic-embed-text` (768-dim)          |
| LLM          | Llama 3.2 (local)                     |
| Language     | Python 3.12                           |

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDFs &    │────▶│  Chunking   │────▶│   FAISS     │
│   Articles  │     │  & Embed    │     │   Index     │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
┌─────────────┐     ┌─────────────┐           │
│   Answer    │◀────│    LLM      │◀──────────┘
│   + Cites   │     │  (Ollama)   │     Semantic Search
└─────────────┘     └─────────────┘
```

## 📁 Project Structure

```
thesis-rag/
├── main.py              # CLI interface
├── ingestion.py         # PDF/web/YouTube processing
├── vector_store.py      # FAISS vector storage
├── qa_chain.py          # LLM query chain with citations
├── requirements.txt
└── data/                # Your research papers (not tracked)
    ├── pdfs/
    ├── web_articles/
    └── vector_store/
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/designer-coderajay/thesis-rag.git
cd thesis-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add Your Papers

Place PDFs in `data/pdfs/` with naming convention:

```
001_authorYEARtitle_YEAR.pdf
# Example: 018_wang2023interpretability_2023.pdf
```

### 3. Index Documents

```bash
python main.py ingest
```

### 4. Query

**CLI:**

```bash
python main.py chat
```

## 💡 Features

### Smart Citation Extraction

Filenames are parsed for metadata:

```
049_bills2023language_2023.pdf
     ↓
Author: Bills, Year: 2023
```

### Source-Grounded Responses

```
Superposition allows neural networks to represent
more features than neurons [Source 1]. This is
demonstrated through toy models [Source 2].

---
SOURCES USED:
[Source 1]: Templeton (2024). Scaling Monosemanticity
[Source 2]: Elhage (2022). Toy Models of Superposition
```

### Multiple Input Types

- ✅ PDF papers (PyPDF2)
- ✅ Web articles (requests + BeautifulSoup)
- ✅ YouTube transcripts (youtube-transcript-api)

## 📊 Performance

| Metric          | Value                      |
| --------------- | -------------------------- |
| Indexed chunks  | ~6,000                     |
| Embedding model | nomic-embed-text (768-dim) |
| Search latency  | <100ms                     |
| Response time   | 30-60s (local Llama)       |

## 🔧 Configuration

### Local LLM (Ollama)

```bash
ollama pull llama3.2
ollama serve
```

## 📝 Thesis Context

This system was built for my Master's thesis:

> **"Explainable AI for LLMs: Causally Grounded Mechanistic Interpretability and Concise Natural-Language Explanations"**

Key research areas covered:

- Mechanistic Interpretability (IOI circuits, superposition)
- Feature Attribution (SHAP, LIME, Integrated Gradients)
- Evaluation Benchmarks (ERASER, e-SNLI)

## 📄 License

MIT License - See [LICENSE](LICENSE) file.

## 🤝 Contributing

This is a personal thesis project. Feel free to fork and adapt for your own research!

---

_Built with ❤️ for explainable AI research_
