<h1 align="center">
  TaxYar (تَکس‌یار)
  <br>
</h1>

<h4 align="center">A Persian-language RAG assistant that answers questions about Iran's Direct Taxes Act, with answers grounded in, and cited to, the actual legal articles.</h4>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#limitations">Limitations</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#license">License</a>
</p>

<!-- TODO: add a screenshot or GIF of the chat UI, e.g. ![screenshot](img/demo.gif) -->

## Overview

Finding the right provision in Iranian tax law is slow: the text is large, heavily structured (chapters, articles, notes) and constantly amended. Keyword search misses the intent of a question, and a plain LLM may produce fluent answers with no legal basis.

**TaxYar** combines semantic retrieval with a grounded language model. It retrieves the most relevant articles of the **Iranian Direct Taxes Act**, hands only that text to the LLM, and returns an answer that references the article number and a link to the official source. If the retrieved context isn't enough, the assistant says so instead of guessing. Questions outside the Direct Taxes Act are politely declined.

> **Current scope:** Direct Taxes Act only. VAT law, executive regulations, circulars and Supreme Tax Council rulings are not covered yet.

> **Disclaimer:** TaxYar is a research prototype, not legal or tax advice. Always verify against the official text.

This project was developed as a bachelor's final project in Computer Engineering at Islamic Azad University, Science and Research Branch (2026), supervised by Dr. Abbas Kouchari.

## Key Features

* **Retrieval-Augmented Generation (RAG)**
  - Semantic search over the law, then answer generation restricted to the retrieved context.
* **Grounded, cited answers**
  - Responses reference the relevant article number and link back to the official source.
  - Insufficient context → the model declines to guess. Out-of-scope questions → explicit refusal.
* **Structure-aware ingestion**
  - Scrapes the JavaScript-rendered official portal, parses the Chapter → Article → Note hierarchy, and keeps metadata (article number, chapter, cross-references, source URL) attached to every chunk.
* **Persian-first model choices**
  - Embedding and LLM selected based on Persian benchmarks (FaMTEB, Open Persian LLM Leaderboard, MIZAN).
* **Streaming chat UI**
  - Persian, right-to-left interface built on [assistant-ui](https://github.com/assistant-ui/assistant-ui), backed by an OpenAI-compatible streaming (SSE) API.
* **Modular architecture**
  - Embedding, vector store, retriever, LLM and API layers are independent, so swapping a model or database doesn't require touching the rest.
* **Built-in retrieval benchmark**
  - Evaluation module reporting Hit@K, Precision@K, MRR, nDCG and article-rank statistics.

## How It Works

```
Official portal ──► loader ──► chunker ──► embeddings ──► Qdrant
                                                            │
User question ──► FastAPI ──► RAG pipeline ──► retriever ◄──┘
                                   │
                          formatters + system prompt
                                   │
                                  LLM ──► streamed, cited answer
```

**1. Data preparation**
- Pages from the official tax regulations portal are rendered with **Playwright** (the content is injected by JavaScript, so plain HTTP crawling doesn't work).
- A recursive **BeautifulSoup** parser walks the nested DOM and produces flat article records with `article_id`, `article_number`, chapter/part info, `text`, `notes`, `cross_references` and `url_reference`.
- Text is split with LangChain's `RecursiveCharacterTextSplitter` (512-token target) using Persian-aware separators (paragraph, line, `.`, `،`, space, ZWNJ).

**2. Embedding & indexing**
- Chunks are embedded with `jina-embeddings-v3` (1024-dim) and stored in a **Qdrant** collection (`law-taxes-direct`, cosine similarity) together with text and metadata as payload.

**3. Retrieval**
- The question is embedded with the same model and the top-*k* (default 5) chunks are retrieved by similarity, each wrapped in a `RetrievedChunk` (text, score, article number, source URL).

**4. Grounded generation**
- Retrieved context, conversation history and the current question are assembled into a prompt. A system prompt defines the role (Direct Taxes Act legal assistant), forces answers to rely only on the context, requires article citations, and sets refusal behavior. Generation parameters (temperature, max length) are kept conservative for stability.

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Scraping / parsing | Playwright, BeautifulSoup |
| Chunking / orchestration | LangChain |
| Embedding model | `jina-embeddings-v3` |
| Vector database | Qdrant |
| LLM | `google/gemma-3-27b-it` (via an OpenAI-compatible provider such as OpenRouter) |
| Backend | FastAPI, SSE streaming |
| Frontend | Next.js, assistant-ui (customized, RTL Persian) |

## Getting Started

> ⚠️ **TODO:** the commands below are placeholders. Replace them with the real entry points, file names and environment variables from your repository.

### Prerequisites

- Python 3.10+ <!-- TODO: confirm version -->
- Node.js 18+ <!-- TODO: confirm version -->
- A running [Qdrant](https://qdrant.tech/) instance (local Docker or cloud)
- API access for the embedding model and an LLM provider

### Installation

```bash
# Clone the repository
git clone https://github.com/PeymanRM/TaxYar.git   # TODO: confirm URL
cd TaxYar

# Backend dependencies
pip install -r requirements.txt                    # TODO: confirm
playwright install chromium

# Frontend dependencies
cd frontend && npm install && cd ..                # TODO: confirm folder name
```

### Configuration

Create a `.env` file:

```env
# TODO: adjust to your actual variable names
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=law-taxes-direct
JINA_API_KEY=your_key
LLM_API_KEY=your_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemma-3-27b-it
RETRIEVAL_TOP_K=5
```

### Build the knowledge base

```bash
# TODO: replace with your real commands/scripts
python -m ingestion.loader      # scrape and parse the law
python -m ingestion.chunker     # split into chunks
python -m ingestion.embeddings  # generate embeddings
python -m ingestion.qdrant_store # index into Qdrant
```

### Run

```bash
# Backend (FastAPI)
uvicorn api.main:app --reload                      # TODO: confirm module path

# Frontend (in another terminal)
cd frontend && npm run dev                         # http://localhost:3000
```

The chat endpoint is `POST /chat/completions` and follows the OpenAI streaming response format. The backend's CORS policy currently allows the dev frontend at `http://localhost:3000`.

## Evaluation

Retrieval quality was measured on **`retrieval_eval_v0.1`**, a hand-built set of **50 questions** about the Direct Taxes Act. Each question maps to exactly one ground-truth article, and the retriever returns the top 5 chunks. Questions are split by tone to test robustness to phrasing:

| Tone | Questions |
| :--- | :--- |
| Natural / conversational | 25 (50%) |
| Neutral | 15 (30%) |
| Formal | 10 (20%) |

### Overall results

| Metric | Score |
| :--- | :--- |
| Hit@1 / Hit@3 / Hit@5 | 72.0% / 92.0% / 96.0% |
| Precision@1 / @3 / @5 | 0.720 / 0.393 / 0.268 |
| Mean Reciprocal Rank (MRR) | 0.829 |
| Mean nDCG@5 | 0.863 |
| Mean / Median rank of correct article | 1.35 / 1.00 |
| Not-found rate (rank > 5) | 4.0% |

### Results by question tone

| Tone | Hit@1 | Hit@3 | Hit@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Natural | 0.68 | 0.84 | 0.92 | 0.78 |
| Neutral | 0.80 | 1.00 | 1.00 | 0.90 |
| Formal | 0.70 | 1.00 | 1.00 | 0.85 |

Conversational phrasing is the hardest case: both questions the retriever missed entirely belong to that group, likely because it's lexically furthest from the legal text.

> Precision@K is reported as a supplementary signal only. Because each question has a single correct article but retrieval happens at chunk level, several chunks of the same article can appear in the top-*k*.

### Running the evaluation

```bash
# TODO: confirm command and dataset path
python -m evaluation.retrieval_evaluation
```

Reports are saved in timestamped folders (`main_report.json`, `all_answer_stats.json`) so different system versions can be compared.

## Limitations

- **Scope:** only the Direct Taxes Act is indexed.
- **Freshness:** the index is a snapshot; keeping it in sync with amendments is not automated.
- **Evaluation:** only the retrieval layer is benchmarked, on a small hand-made set of single-article questions. Answer quality (faithfulness, correctness) has not been formally evaluated.
- **No metadata filtering/reranking** in the main retrieval path yet.

## Roadmap

- [ ] Expand coverage: VAT law, executive regulations, circulars, Supreme Tax Council rulings
- [ ] Topic tags and richer metadata for filtered retrieval
- [ ] Document-summary indexing for broad questions
- [ ] Hybrid search (semantic + BM25)
- [ ] LLM-driven self-querying for automatic metadata filters
- [ ] Reranking, tested on a harder evaluation set
- [ ] End-to-end generation evaluation (e.g. with RAGAS)
- [ ] Persistent chat history and user accounts (MongoDB/PostgreSQL)

## Acknowledgments

- [jina-embeddings-v3](https://arxiv.org/abs/2409.10173) and [FaMTEB](https://aclanthology.org/2025.findings-emnlp.614/) for Persian embedding selection
- [Gemma 3](https://arxiv.org/abs/2503.19786) by Google
- [Qdrant](https://qdrant.tech/), [LangChain](https://www.langchain.com/), [FastAPI](https://fastapi.tiangolo.com/), [assistant-ui](https://github.com/assistant-ui/assistant-ui)
- Legal text sourced from the Iranian National Tax Administration's [regulations portal](https://regulation.tax.gov.ir/)
- Chunk-size choice informed by Bourbour Hosseinbeigi et al., *Advancing Retrieval-Augmented Generation for Persian* (2025)

## License

<!-- TODO: choose a license, e.g. MIT -->
MIT

---

> LinkedIn [@PeymanRM](https://www.linkedin.com/in/peymanrm) &nbsp;&middot;&nbsp;
> GitHub [@PeymanRM](https://github.com/PeymanRM) &nbsp;&middot;&nbsp;
> X [@Peyman_RM](https://x.com/Peyman_RM)
